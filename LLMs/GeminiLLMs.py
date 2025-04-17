from utils.tools import Tools
class Gemini:
    def __init__(self, config_path):
        config = Tools.load_json(config_path)

gemini= Gemini("./Config/LLMsConfig.json")