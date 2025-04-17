import os

from loguru import logger
os.makedirs("./logs", exist_ok=True)
logger.add("./logs/file_{time}.log")