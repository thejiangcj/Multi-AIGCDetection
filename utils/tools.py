import json
import hashlib
import re
import os
import base64

from .qwen2_5_utils import *

class Tools:
    extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.mp4', '.gif']
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
    def to_json(data_results, to_path):
        # 保存到 JSON 文件
        dir_name = os.path.dirname(to_path)
        
        basename = os.path.join(dir_name,os.path.splitext(os.path.basename(to_path))[0] + ".json")

        with open(basename, 'w', encoding='utf-8') as json_file:
            json.dump(data_results, json_file, ensure_ascii=False, indent=2)

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
        first_think_pos = text.find('<think>')
        last_think_pos = text.rfind('</think>')

        if first_think_pos != -1 and last_think_pos != -1:
            # 提取匹配到的内容
            start = first_think_pos + len('<think>')
            end = last_think_pos
            return text[start:end].strip()
        elif first_think_pos != -1 and last_think_pos == -1 and text.rfind('<conclusion>') !=-1:
          start = first_think_pos + len('<think>')
          end = text.rfind('<conclusion>')
          return text[start:end].strip()
        return ''

    @staticmethod
    def extract_conclusion_content(text):
        """提取conclusion标签内容（带缓存）"""
        first_think_pos = text.find('<think>')
        last_think_pos = text.rfind('</think>')

        if first_think_pos != -1 and last_think_pos != -1:
            # 提取匹配到的内容
            start = first_think_pos + len('<think>')
            end = last_think_pos + len("</think>")
            match = re.search(r'<conclusion>(.*?)</conclusion>', text[end:], re.DOTALL)
            return match.group(1).strip() if match else ''
        elif first_think_pos != -1 and last_think_pos == -1 and text.rfind('<conclusion>') !=-1:

          start = first_think_pos + len('<think>')
          end = text.rfind('<conclusion>')
          match = re.search(r'<conclusion>(.*?)</conclusion>', text[end:], re.DOTALL)
          return match.group(1).strip() if match else ''
        return ''

    @staticmethod
    def get_think_tag_directly(text):
        """直接获取think标签内容"""
        first_think_pos = text.find('<think>')
        last_think_pos = text.rfind('</think>')

        if first_think_pos != -1 and last_think_pos != -1:
            # 提取匹配到的内容
            start = first_think_pos + len('<think>')
            end = last_think_pos + len("</think>")
            return text[first_think_pos:end].strip()
        return ''

    @staticmethod
    def conclusion_has_single_word(text):
        """检查conclusion内容是否只有一个单词"""
        content = Tools.extract_conclusion_content(text)
        return True if "fake" in content or "real" in content else False

    @staticmethod
    def label_vs(text, label):
        """检查conclusion内容是否只有一个单词"""
        # text = Tools.extract_conclusion_content(text)
        to_label = 0
        if text.strip().lower() == label.strip().lower():
            to_label = 1
        elif f"is {label}" in text.strip().lower():
            to_label = 1
        elif text.strip().startswith(label) or text.strip().endswith(label):
            to_label = 1
        elif f"as {label}" in text.strip().lower() or f"as '{label}'" in text.strip().lower():
            to_label = 1
        elif "synthetic generation" in text and label=="fake":
            to_label = 1
        elif "real capture" in text and label=="real":
            to_label = 1

        return to_label
    
    @staticmethod
    def detect_Type_from_path(file_path):
        """根据文件路径检测文件类型"""
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
        video_extensions = ['.mp4', '.gif']
        file_type = None
        file_extension = os.path.splitext(file_path)[1].lower()
        if file_extension in image_extensions:
            file_type = "image"
        elif file_extension in video_extensions:
            file_type = "video"
        return file_type
    
    @staticmethod
    def encoder_file_path(file_path):
        """将图片编码为base64格式"""
        file_base64 = Tools.encode_image(file_path)
        fps_list = None
        if Tools.detect_Type_from_path(file_path) == "image":
            file_base64 = f"data:image/jpeg;base64,{file_base64}"
        elif Tools.detect_Type_from_path(file_path) == "video":
            file_base64, fps_list = local_video_to_base64(file_path)
        else:
            file_base64 = None
        return file_base64, Tools.detect_Type_from_path(file_path), fps_list

    @staticmethod
    def get_label(file_path):
        """获取文件类型"""
        file_type = None
        if "fake" in file_path:
            file_type = "fake"
        elif "real" in file_path:
            file_type = "real"
        return file_type
    
    @staticmethod
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")