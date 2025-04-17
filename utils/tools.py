import json

class Tools:
    @staticmethod
    def load_prompt(self, prompt_path):
        with open(prompt_path, 'r', encoding='utf-8') as file:
            return file.read(
                prompt_path
            )
    @staticmethod
    def load_json(self, json_path):
        with open(json_path, 'r', encoding='utf-8') as file:
            return json.load(
                json_path
            )