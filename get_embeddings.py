import argparse
from collections import defaultdict
import json
import logging
import math
import os
import sys
import queue
from typing import Dict, List, Optional, Union

from tqdm.autonotebook import trange
# import datasets
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
from transformers import AutoModel, AutoTokenizer
from transformers import AutoModelForCausalLM

from huggingface_hub import login
login(token = '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')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s : %(message)s'
)

logger = logging.getLogger('eval_mteb_qwen.py')

def get_detailed_instruct(task_description: str) -> str:
    if not task_description:
        return ''

    return 'Instruct: {}\nQuery: '.format(task_description)


class Encoder(torch.nn.Module):
    def __init__(self, name_or_path:str, pooling: str):
        super().__init__()
        self.model = AutoModel.from_pretrained(name_or_path, trust_remote_code=True)#, cache_dir='/10TBdrive/pooja/ai-mitigations/.cache/huggingface/')
        self.model = self.model#.half()
        self.model.eval()  
        self.pooling = pooling

    def forward(self, **features) -> torch.Tensor:
        output = self.model(**features, output_hidden_states=True, return_dict=True)
        hidden_state = output.hidden_states[-1]        
        embeddings = self.pooler(hidden_state, **features)
        return embeddings

    def pooler(
        self,
        hidden_state: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        if attention_mask.ndim == 2:
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_state.size())
        elif attention_mask.ndim == 3:
            mask_expanded = attention_mask
        else:
            raise RuntimeError(f"Unexpected {attention_mask.ndim=}")

        hidden_state = hidden_state * mask_expanded

        if self.pooling == 'first':
            pooled_output = hidden_state[:, 0]

        elif self.pooling == 'last':
            left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
            if left_padding:
                return hidden_state[:, -1]
            else:
                sequence_lengths = attention_mask.sum(dim=1) - 1
                batch_size = hidden_state.shape[0]
            return hidden_state[torch.arange(batch_size, device=hidden_state.device), sequence_lengths]
        elif self.pooling == 'mean':
            # TODO: weight
            lengths = mask_expanded.sum(1).clamp(min=1e-9)
            pooled_output = hidden_state.sum(dim=1) / lengths

        elif self.pooling == 'weightedmean':
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_state.size()).float()
            # hidden_state shape: bs, seq, hidden_dim
            weights = (
                    torch.arange(start=1, end=hidden_state.shape[1] + 1)
                    .unsqueeze(0)
                    .unsqueeze(-1)
                    .expand(hidden_state.size())
                    .float().to(hidden_state.device)
                )
            assert weights.shape == hidden_state.shape == input_mask_expanded.shape
            input_mask_expanded = input_mask_expanded * weights

            sum_embeddings = torch.sum(hidden_state * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            pooled_output = sum_embeddings / sum_mask

        else:
            raise ValueError(f"Wrong pooler mode : {self.pooling}")
        return pooled_output


class Wrapper:
    def __init__(
        self,
        tokenizer,
        encoder: Encoder,
        batch_size: int,
        max_seq_len: int = 512,
        normalize_embeddings: bool = False,
        default_query: bool = False,
        force_default: bool = False,
        sep: str = " ",
        mp_tensor_to_cuda: bool = False,
        instruction: str = None,
        attn_type: str = None
    ):
        self.tokenizer = tokenizer
        self.model = encoder
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.pool: dict = None
        self.normalize_embeddings = normalize_embeddings
        self.mp_tensor_to_cuda = mp_tensor_to_cuda
        self._target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.eod_id = self.tokenizer.convert_tokens_to_ids("<|endoftext|>")
        self.instruction = instruction
        self.default_query = default_query
        self.force_default = force_default
 
        if self.tokenizer.padding_side != 'right':
            logger.warning(f"Change tokenizer.padding_side from {self.tokenizer.padding_side} to right")
            self.tokenizer.padding_side = 'right'
        if self.tokenizer.pad_token is None:
            logger.warning(f"Set tokenizer.pad_token as eos_token {self.tokenizer.eos_token}")
            self.tokenizer.pad_token='<|endoftext|>'

    def start(self, target_devices: Optional[List[str]] = None):
        """
        Starts multi process to process the encoding with several, independent processes.
        This method is recommended if you want to encode on multiple GPUs. It is advised
        to start only one process per GPU. This method works together with encode_multi_process
        :param target_devices: PyTorch target devices, e.g. cuda:0, cuda:1... If None, all available CUDA devices will be used
        :return: Returns a dict with the target processes, an input queue and and output queue.
        """
        if target_devices is None:
            if torch.cuda.is_available():
                target_devices = ['cuda:{}'.format(i) for i in range(torch.cuda.device_count())]
            else:
                logger.info("CUDA is not available. Start 4 CPU worker")
                target_devices = ['cpu']*4

        logger.info("Start multi-process pool on devices: {}".format(', '.join(map(str, target_devices))))
        print('multi instruction', self.instruction)
        ctx = mp.get_context('spawn')
        input_queue = ctx.Queue()
        output_queue = ctx.Queue()
        processes = []

        for cuda_id in target_devices:
            p = ctx.Process(
                target=self._encode_multi_process_worker,
                args=(cuda_id, self, input_queue, output_queue),
                daemon=True
            )
            p.start()
            processes.append(p)

        self.pool = {'input': input_queue, 'output': output_queue, 'processes': processes}

    def stop(self):
        """
        Stops all processes started with start_multi_process_pool
        """
        for p in self.pool['processes']:
            p.terminate()

        for p in self.pool['processes']:
            p.join()
            p.close()

        self.pool['input'].close()
        self.pool['output'].close()

    @staticmethod
    def _encode_multi_process_worker(target_device: str, model, input_queue, results_queue):
        """
        Internal working process to encode sentences in multi-process setup
        """
        while True:
            try:
                id, sentences, kwargs = input_queue.get()
                kwargs.update(device=target_device, show_progress_bar=False, convert_to_numpy=True)
                embeddings = model._encode(sentences, **kwargs)
                results_queue.put([id, embeddings])
            except queue.Empty:
                break

    def encode_multi_process(
        self,
        sentences: List[str],
        **kwargs
    ):
        """
        This method allows to run encode() on multiple GPUs. The sentences are chunked into smaller packages
        and sent to individual processes, which encode these on the different GPUs. This method is only suitable
        for encoding large sets of sentences
        :param sentences: List of sentences
        :param pool: A pool of workers started with SentenceTransformer.start_multi_process_pool
        :param chunk_size: Sentences are chunked and sent to the individual processes. If none, it determine a sensible size.
        :param kwargs: other keyword arguments for model.encode() such as batch_size
        :return: Numpy matrix with all embeddings
        """
        part_size = math.ceil(len(sentences) / len(self.pool["processes"]))
        chunk_size = part_size if part_size < 3200 else 3200  # for retrieval chunk 50000

        logger.debug(f"Chunk data into {math.ceil(len(sentences) / chunk_size)} packages of size {chunk_size}")

        input_queue = self.pool['input']
        last_chunk_id = 0
        chunk = []

        for sentence in sentences:
            chunk.append(sentence)
            if len(chunk) >= chunk_size:
                input_queue.put([last_chunk_id, chunk, kwargs])
                last_chunk_id += 1
                chunk = []

        if len(chunk) > 0:
            input_queue.put([last_chunk_id, chunk, kwargs])
            last_chunk_id += 1

        output_queue = self.pool['output']
        results_list = sorted([output_queue.get() for _ in range(last_chunk_id)], key=lambda x: x[0])
        embeddings = np.concatenate([result[1] for result in results_list])
        return embeddings

    @staticmethod
    def batch_to_device(batch, target_device):
        """
        send a pytorch batch to a device (CPU/GPU)
        """
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(target_device)
        return batch

    def _text_length(self, text: Union[List[int], List[List[int]]]):
        """
        Help function to get the length for the input text. Text can be either
        a list of ints (which means a single text as input), or a tuple of list of ints
        (representing several text inputs to the model).
        """

        if isinstance(text, dict):              #{key: value} case
            return len(next(iter(text.values())))
        elif not hasattr(text, '__len__'):      #Object has no len() method
            return 1
        elif len(text) == 0 or isinstance(text[0], int):    #Empty string or list of ints
            return len(text)
        else:
            return sum([len(t) for t in text])      #Sum of length of individual strings

    def _tokenize(self, sentences: List[str], is_query: bool):
        
        batch_dict = self.tokenizer(sentences, max_length=self.max_seq_len - 1, return_attention_mask=True, padding=True, truncation=True, return_tensors="pt")
        # batch_dict['input_ids'] = [input_ids + [self.tokenizer.eos_token_id] for input_ids in batch_dict['input_ids']]
        # batch_dict = self.tokenizer.pad(batch_dict, padding=True, truncation=True, return_attention_mask=True, return_tensors='pt')
        # batch_dict['is_causal'] = False
        return batch_dict


    def _encode(
        self,
        sentences: List[str],
        is_query: bool,
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
        device: str = None,
        show_progress_bar: bool = True,
        **kwargs
    ):
        """
        Computes sentence embeddings
        :param sentences: the sentences to embed
        :param batch_size: the batch size used for the computation
        :param show_progress_bar: Output a progress bar when encode sentences
        :param output_value:  Default sentence_embedding, to get sentence embeddings. Can be set to token_embeddings to get wordpiece token embeddings. Set to None, to get all output values
        :param convert_to_numpy: If true, the output is a list of numpy vectors. Else, it is a list of pytorch tensors.
        :param convert_to_tensor: If true, you get one large tensor as return. Overwrites any setting from convert_to_numpy
        :param device: Which torch.device to use for the computation
        :param normalize_embeddings: If set to true, returned vectors will have length 1. In that case, the faster dot-product (util.dot_score) instead of cosine similarity can be used.
        :return:
           By default, a list of tensors is returned. If convert_to_tensor, a stacked tensor is returned. If convert_to_numpy, a numpy matrix is returned.
        """
        self.model.eval()

        if convert_to_tensor:
            convert_to_numpy = False

        input_was_string = False
        if isinstance(sentences, str) or not hasattr(sentences, '__len__'): #Cast an individual sentence to a list with length 1
            sentences = [sentences]
            input_was_string = True

        if device is None:
            device = self._target_device

        self.model.to(device)

        all_embeddings = []
        length_sorted_idx = np.argsort([-self._text_length(s) for s in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

        for start_index in trange(0, len(sentences), self.batch_size, desc="Batches", disable=not show_progress_bar):
            sentences_batch = sentences_sorted[start_index:start_index + self.batch_size]
            features = self._tokenize(sentences_batch, is_query)
            features = self.batch_to_device(features, device)

            with torch.no_grad():
                embeddings = self.model(**features)
                # embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

                if self.normalize_embeddings:
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

                # fixes for #522 and #487 to avoid oom problems on gpu with large datasets
                if convert_to_numpy:
                    embeddings = embeddings.cpu()

                all_embeddings.extend(embeddings)

        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]

        if convert_to_tensor:
            all_embeddings = torch.stack(all_embeddings)
        elif convert_to_numpy:
            #all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])
            all_embeddings = np.asarray([emb.to(torch.float).numpy() for emb in all_embeddings])
        if input_was_string:
            all_embeddings = all_embeddings[0]

        return all_embeddings

    def encode(
        self,
        sentences: List[str],
        is_query: Optional[bool] = None,
        convert_to_tensor: bool = False,
        **kwargs
    ):
        is_query = self.default_query if is_query is None else is_query
        if is_query and self.instruction:
           sentences = [self.instruction + sent for sent in sentences]
        kwargs.update(is_query=is_query)
        if self.pool is not None:
            kwargs.update(show_progress_bar=False)
            embeddings = self.encode_multi_process(sentences, **kwargs)
            if convert_to_tensor:
                embeddings = torch.from_numpy(embeddings)
                if self.mp_tensor_to_cuda and torch.cuda.is_available():
                    embeddings = embeddings.to(torch.device('cuda'))  # default 0-th gpu
            return embeddings

        return self._encode(sentences, convert_to_tensor=convert_to_tensor, **kwargs)

    def encode_queries(self, queries: List[str], **kwargs):
        is_query = self.default_query if self.force_default else True
        return self.encode(queries, is_query=is_query, **kwargs)

    def encode_corpus(self, corpus: List[Dict[str, str]], **kwargs):
        # borrowed from mteb.abstasks.AbsTaskRetrieval.DRESModel
        if type(corpus) is dict:
            sentences = [
                (corpus["title"][i] + self.sep + corpus["text"][i]).strip()
                if "title" in corpus
                else corpus["text"][i].strip()
                for i in range(len(corpus["text"]))
            ]
        elif isinstance(corpus[0], dict):
            sentences = [
                (doc["title"] + self.sep + doc["text"]).strip() if "title" in doc else doc["text"].strip()
                for doc in corpus
            ]
        else:
            sentences = corpus
        is_query = self.default_query if self.force_default else False
        return self.encode(sentences, is_query=is_query, **kwargs)

def get_embeddings(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)#, cache_dir='/10TBdrive/pooja/ai-mitigations/.cache/huggingface/')
    encoder = Encoder(args.model, args.pooling)
    model = Wrapper(
        tokenizer, encoder,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        normalize_embeddings=args.norm
    )
    instruction = get_detailed_instruct("Given a query, retrieve model descriptions that are semantically similar to the model description in the given query")
    model.instruction = get_detailed_instruct(instruction)

    embeddings_corpus = model.encode_corpus(args.corpus)
    embeddings_queries = model.encode_queries(args.queries)

    return embeddings_corpus, embeddings_queries

def get_embeddings_queries(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)#, cache_dir='/10TBdrive/pooja/ai-mitigations/.cache/huggingface/')
    encoder = Encoder(args.model, args.pooling)
    model = Wrapper(
        tokenizer, encoder,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        normalize_embeddings=args.norm
    )
    instruction = get_detailed_instruct("Given a query, retrieve model descriptions that are semantically similar to the model description in the given query")
    model.instruction = get_detailed_instruct(instruction)

    embeddings_queries = model.encode_queries(args.queries)

    return embeddings_queries

# get embeddings computed from get_embeddings function or load from file
def compute_embeddings(corpus, queries, model, max_len, batch_size, embeddings_file=None):
    _PARSER = argparse.ArgumentParser()
    _PARSER.add_argument(
        "-m", "--model", type=str, default='BAAI/bge-large-en-v1.5'
    )
    _PARSER.add_argument("--pooling", type=str, default='last')
    _PARSER.add_argument("--output_dir", type=str, default=None)
    _PARSER.add_argument("--default_type", type=str, default='query')
    _PARSER.add_argument("--max_seq_len", type=int, default=4096)
    _PARSER.add_argument("-b", "--batch_size", type=int, default=32)
    _PARSER.add_argument(
        "-t", "--task", type=str, default=None  # None for running default tasks
    )
    _PARSER.add_argument("--norm", action="store_true")
    _PARSER.add_argument("-q", "--queries", type=list, default=None)
    _PARSER.add_argument("-c", "--corpus", type=list, default=None)

    _ARGS = _PARSER.parse_args()

    _ARGS.model = model
    _ARGS.queries = queries
    _ARGS.corpus = corpus
    _ARGS.batch_size = batch_size
    _ARGS.max_seq_len = max_len
    _ARGS.embeddings_file = embeddings_file

    if os.path.isfile("{}_{}.npy".format(embeddings_file, model.split('/')[-1])):
        embeddings_corpus = np.load("{}_{}.npy".format(embeddings_file, model.split('/')[-1]))
        embeddings_queries = get_embeddings_queries(_ARGS)
    else:   
        embeddings_corpus, embeddings_queries = get_embeddings(_ARGS)
        np.save("from_server/new_data/embeddings/embedding_corpus_{}_{}.npy".format(embeddings_file, model.split('/')[-1]), embeddings_corpus)
        np.save("from_server/new_data/embeddings/embedding_queries_{}_{}.npy".format(embeddings_file, model.split('/')[-1]), embeddings_queries)

    return embeddings_corpus, embeddings_queries
    
    
    
#     embeddings_corpus, embeddings_queries = main(_ARGS)
#     return embeddings_corpus, embeddings_queries

if __name__ == "__main__":
    pass

