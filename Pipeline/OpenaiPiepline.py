import sys, os

# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from openai import OpenAI
from utils.qwen2_5_utils import *
from utils.tools import Tools
from LLMs.OpenaiLLMs import openaiLLMs
from LLMs.Qwen2_5_VL_LLMs import Qwen2_5_VL_LLMs

class OpenaiPipeline:
    def __init__(self, llm_config):
        self.api = llm_config["Openai"]["APIKey"]
        self.url = llm_config["Openai"]["url"]
        self.model_name = llm_config["Openai"]["model_name"]
        self.client = OpenAI(base_url=self.url, api_key=self.api)

        self.DP_system_prompt = "You are a AIGC Detector. Determine whether the following {{Type}} is AI-generated (fake) or human-produced (real). Provide your judgment clearly enclosed in <conclusion>fake or real</conclusion>. The conclusion is enclosed within <conclusion> </conclusion> tags, <conclusion> real or fake </conclusion>. <conclusion> content must strictly be (real/fake)."
        self.DP_user_prompt = "Is the following {{Type}} fake or real?"
    def pipeline_with_single_file(self, file_path, prompt_method="DP",**kwargs):
        file_type = Tools.detect_Type_from_path(file_path)
        label = Tools.get_label(file_path)

        system_prompt, user_prompt = self.get_prompt(prompt_method, file_type=file_type)
        if file_type == "image":
            res = self.pipeline_with_image(file_path, file_type, system_prompt, user_prompt)
        elif file_type == "video":
            res = self.pipeline_with_video(file_path, file_type, system_prompt, user_prompt)
        else:
            pass
        to_item = {
            "file_path": file_path,
            "label": label,
            "file_type": file_type,
            "prompt_method": prompt_method,
            "model": self.model_name,
            "raw_predict": res,
            "predict": ""
        }
        to_item["predict"] = Tools.get_think_tag_directly(to_item["raw_predict"])
        return to_item
    def pipeline(self, data_path, to_path, prompt_method="DP"):
        results = []
        for root, dirs, files in os.walk(data_path):
            for file in files:
                file_path = os.path.join(root, file)
                if Tools.detect_Type_from_path(file_path):
                    res = self.pipeline_with_single_file(file_path, prompt_method=prompt_method)
                    results.append(res)

                else:
                    print(f"Unsupported file type: {file_path}")

                if len(results) % 50 == 0:
                    Tools.to_json(results, to_path)
        Tools.to_json(results, to_path)

    def pipeline_with_image(self, file_path, file_type, system_prompt, user_prompt, **kwargs):
        file_base64 = Tools.encode_image(file_path)
        messages = [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": user_prompt,
                        },
                        {
                            "type": f"{file_type}_url",
                            f"{file_type}_url": {
                                "url": file_base64
                            },
                        },
                    ],
                }
            ]
        response = self.infer(messages)
        return response

    def pipeline_with_video(self, file_path, file_type, system_prompt, user_prompt, **kwargs):
        if "Qwen2.5-VL" in self.model_name:
            messages, args = Qwen2_5_VL_LLMs.get_messages_with_video(file_path, system_prompt, user_prompt)
            extra_body={
                    "mm_processor_kwargs": args
                }
            response = openaiLLMs.infer(messages, extra_body=extra_body)
        return response

    def get_prompt(self, prompt_method, **kwargs):
        if prompt_method == "DP":
            system_prompt = self.DP_system_prompt.replace("{{Type}}", kwargs.get("file_type", "image"))
            user_prompt = self.DP_user_prompt.replace("{{Type}}", kwargs.get("file_type", "image"))
        else:
            raise ValueError(f"Unknown prompt method: {prompt_method}")
        return system_prompt, user_prompt

openaiPipeline = OpenaiPipeline(Tools.load_json("./Config/LLMsConfig.json"))

if __name__ == "__main__":
   
    openaiPipeline.pipeline(
        data_path="/raid/share/jiangchangjiang/aws/dataset/fakeclue/train",
        to_path="./data/test.json",
        prompt_method="DP"
    )