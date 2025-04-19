from utils.tools import Tools

from google import genai
from google.genai import types
import os
import time

class Gemini:
    def __init__(self, config_path, prompt_path):
        config = Tools.load_json(config_path)
        prompt_config = Tools.load_json(prompt_path)
        api = config["Gemini"]["APIKey"]
        self.client = genai.Client(api_key=api)

        self.image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
        self.video_extensions = {'.mp4'}

        self.system_prompt_map = {
            "image": Tools.load_prompt(prompt_config["DistilPrompt"]["image"]),
            "video": Tools.load_prompt(prompt_config["DistilPrompt"]["video"])
        }

    def infer_with_file(self, file_path, **kwargs):
        Type = "image" if self.is_image_file(file_path) else "video"
        Label = kwargs.get("label")
        system_prompt = self.system_prompt_map[Type]

        uploaded_file = self.client.files.upload(file=file_path)
        generate_content_config = types.GenerateContentConfig(
            response_mime_type="text/plain",
            system_instruction=[
                types.Part.from_text(text=system_prompt),
            ],
        )
        while uploaded_file.state.name == "PROCESSING":
            print("processing video...")
            time.sleep(5)
            uploaded_file = self.client.files.get(name=uploaded_file.name)

        result = self.client.models.generate_content(
            # model="gemini-2.0-flash", # 2.0 flash thinking
            model="gemini-2.0-flash-thinking-exp-01-21",
            contents=[uploaded_file, f"This {Type} is {Label}, explain the reason."],
            config=generate_content_config
        )
        return result.text


    def is_image_file(self,filename):
        """
        判断一个文件是否是图片格式（根据文件后缀名）
        支持的图片格式包括：.jpg, .jpeg, .png, .gif, .bmp, .tiff, .webp
        """
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
        ext = os.path.splitext(filename)[1].lower()
        return ext in image_extensions

    def is_video_file(self,filename):
        """
        判断一个文件是否是图片格式（根据文件后缀名）
        支持的图片格式包括：.jpg, .jpeg, .png, .gif, .bmp, .tiff, .webp
        """
        image_extensions = {'.mp4'}
        ext = os.path.splitext(filename)[1].lower()
        return ext in image_extensions

gemini= Gemini("./Config/LLMsConfig.json", prompt_path="./Config/PromptsConfig.json")