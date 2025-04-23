import json
import hashlib
import re
import os
from loguru import logger

api = "AIzaSyABAiY59RAZiDCPCGXezHp_Y3_w3gIpzfY" ## 这里写API

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
        # text = Tools.extract_conclusion_content(text)
        return 1 if text.strip().lower() == label.strip().lower() else 0

from google import genai
from google.genai import types
import os
import time
import traceback

class GeminiPipeline:
    def __init__(self):
        self.search_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', 'mp4']
        prompt_config = {
            "image": "/content/drive/MyDrive/Project/AIGC-Detection/Prompts/ImageInferWithLabelPrompt.txt",
            "video": "/content/drive/MyDrive/Project/AIGC-Detection/Prompts/VideoInferWithLabelPrompt.txt"
        }
        self.client = genai.Client(api_key=api)

        self.image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
        self.video_extensions = {'.mp4'}

        self.system_prompt_map = {
            "image": Tools.load_prompt(prompt_config["image"]),
            "video": Tools.load_prompt(prompt_config["video"])
        }

    def infer_with_file(self, file_path, **kwargs):
        # Type = "image" if self.is_image_file(file_path) else "video"
        # Label = kwargs.get("label")
        # system_prompt = self.system_prompt_map[Type]

        uploaded_file = self.client.files.upload(file=file_path)
        generate_content_config = types.GenerateContentConfig(
            response_mime_type="text/plain",
            system_instruction=[
                types.Part.from_text(text=kwargs["system_prompt"]),
            ],
        )
        while uploaded_file.state.name == "PROCESSING":
            print("processing video...")
            time.sleep(5)
            uploaded_file = self.client.files.get(name=uploaded_file.name)

        result = self.client.models.generate_content(
            # model="gemini-2.0-flash", # 2.0 flash thinking
            model="gemini-2.0-flash-thinking-exp-01-21",
            contents=[uploaded_file, kwargs["user_prompt"]],
            config=generate_content_config
        )
        return result.text

    def _search_raw_dataset(self, file_path):
        for i in self.raw_datasets:
            if i in file_path:
                return i
        return "FakeThread"

    def is_image_file(self,filename):
        """
        判断一个文件是否是图片格式（根据文件后缀名）
        支持的图片格式包括：.jpg, .jpeg, .png, .gif, .bmp, .tiff, .webp
        """
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
        ext = os.path.splitext(filename)[1].lower()
        return ext in image_extensions

    def _llm_infer(self, llm_engine, file_path, **kwargs):
        result = ""

        if llm_engine == "gemini":
            result = gemini.infer_with_file(file_path, system_prompt=kwargs["system_prompt"], user_prompt=kwargs["user_prompt"])
        return result

    def is_video_file(self,filename):
        """
        判断一个文件是否是图片格式（根据文件后缀名）
        支持的图片格式包括：.jpg, .jpeg, .png, .gif, .bmp, .tiff, .webp
        """
        image_extensions = {'.mp4'}
        ext = os.path.splitext(filename)[1].lower()
        return ext in image_extensions

    def pipeline(self, to_path, raw_data_path, llm_engine="gemini", **kwargs):
        assert to_path.endswith(".jsonl")

        if os.path.exists(to_path):
            to_data = Tools.load_jsonl(to_path)
            uids = [i["id"] for i in to_data]
            uid_map = {i["id"]:i for i in to_data}
        else:
            uids = []
            to_data = []

        for root, dirs, files in os.walk(raw_data_path):
            for file in files:
                file_path = os.path.join(root, file)
                if os.path.splitext(file)[-1] in self.search_extensions:
                    rel_path = os.path.relpath(file_path, raw_data_path)
                    # rwa_dataset = self._search_raw_dataset(file_path)
                    # category = self._search_category(file_path)
                    uid = Tools.generate_id(rel_path)
                    if uid in uids and "429 RESOURCE_EXHAUSTED" not in uid_map[uid]["raw_predict"]:
                        print(f"uid:{uid},\nrel_path:{rel_path}已经存在")
                        continue
                    Type = "image" if self.is_image_file(file_path) else "video"
                    Label = "real" if "real" in file_path else "fake"
                    system_prompt = self.system_prompt_map[Type]
                    user_prompt = f"This {Type} is {Label}, explain the reason."
                    to_dict = {
                        "id": uid,
                        "label": Label,
                        "type": Type,
                        "file_path": rel_path,
                        "raw_dataset": "",
                        "raw_prompt": system_prompt + "\n\n\n" + user_prompt,
                        "category": "",
                        "error": 0,
                        "raw_predict": "",
                        "think": "",
                        "conclusion": "",
                        "match": 0
                    }
                    result,think,conclusion,matchs,error = "","","", 0,0
                    try:
                        result = self._llm_infer(llm_engine=llm_engine, file_path=file_path, system_prompt=system_prompt, user_prompt=user_prompt)
                        think = Tools.extract_think_content(result)
                        conclusion = Tools.extract_conclusion_content(result)
                        matchs = Tools.label_vs(conclusion, to_dict["label"])
                        if "429 RESOURCE_EXHAUSTED" in str(result):
                            logger.error(str(result))
                            break
                    except Exception as E:
                        traceback.print_exc()
                        logger.error(f"Error is {str(E)}")
                        result = str(E)
                        error = 1


                    to_dict["raw_predict"] = result
                    to_dict["think"] = think
                    to_dict["conclusion"] = conclusion
                    to_dict["match"] = matchs
                    to_dict["error"] = error

                    to_data.append(to_dict)
                    logger.info(f"已经处理了{len(to_data)}条数据。")
                    Tools.to_file(to_data, to_path)

if __name__ == "__main__":
    gemini = GeminiPipeline()

    gemini.pipeline(
        to_path="/content/drive/MyDrive/Project/AIGC-Detection/data/fakethread/image/gemini.jsonl",
        raw_data_path="/content/drive/MyDrive/Project/AIGC-Detection/data/fakethread/image",
        llm_engine="gemini"
    )
