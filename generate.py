import json
from copy import deepcopy
from openai import OpenAI

class RiskGenerator:
    def __init__(self, api_key, model="gpt-4o"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def format_prompt(self, card_id, section):
        """Format the prompt for risk generation."""
        system_content = open("prompts/system_prompt.txt", "r").read()
        user_content = open("prompts/user_prompt.txt", "r").read()
        user_content = user_content.format(str(card_id), section)
        prompt = [
            { 'role': 'system', 'content': system_content },
            { 'role': 'user', 'content': user_content }
        ]
        return prompt

    def generate_risks(self, card_id, section):
        """Generate risks using OpenAI."""
        prompt = self.format_prompt(card_id, section)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=prompt
        )
        return json.loads(response.choices[0].message.content.strip())