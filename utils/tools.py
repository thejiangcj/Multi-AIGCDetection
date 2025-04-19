import json
import hashlib
import re
import os

class Tools:
    @staticmethod
    def to_file(data, to_path):
        with open(to_path, 'w', encoding='utf-8') as jsonl_file:
            for entry in data:
                jsonl_file.write(json.dumps(entry) + '\n')
        # 保存到 JSON 文件
        dir_name = os.path.dirname(to_path)
        basename = os.path.join(dir_name,os.path.splitext(os.path.basename(to_path))[0] + ".json")
        with open(basename, 'w', encoding='utf-8') as json_file:
            json.dump(data, json_file, ensure_ascii=False, indent=2)

    @staticmethod
    def load_prompt(prompt_path):
        with open(prompt_path, 'r', encoding='utf-8') as file:
            return file.read()
    @staticmethod
    def load_json(json_path):
        with open(json_path, 'r', encoding='utf-8') as file:
            return json.load(
                file
            )
    @staticmethod  
    def generate_id(text: str) -> str:
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    
    @staticmethod
    def load_jsonl(jsonl_path):
        with open(jsonl_path, 'r', encoding='utf-8') as file:
            return [json.loads(line) for line in file]
    @staticmethod
    def has_think_tags(text):
        """检查是否存在<pros>标签"""
        return '<think>' in text and '</think>' in text
    
    @staticmethod
    def has_conclusion_tags(text):
        """检查是否存在<conclusion>标签"""
        return '<conclusion>' in text and '</conclusion>' in text

    @staticmethod
    def extract_think_content(text):
        """提取think标签内容（带缓存）"""
        match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
        return match.group(1).strip() if match else ''

    @staticmethod
    def extract_conclusion_content(text):
        """提取conclusion标签内容（带缓存）"""
        match = re.search(r'<conclusion>(.*?)</conclusion>', text, re.DOTALL)
        return match.group(1).strip() if match else ''

    @staticmethod
    def conclusion_has_single_word(text):
        """检查conclusion内容是否只有一个单词"""
        content = Tools.extract_conclusion_content(text)
        return True if "fake" in content or "real" in content else False

    @staticmethod
    def label_vs(text, label):
        """检查conclusion内容是否只有一个单词"""
        # print(text)
        text = Tools.extract_conclusion_content(text)
        return 1 if text.strip().lower() == label.strip().lower() else 0
    
    