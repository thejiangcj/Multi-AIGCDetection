import time
from functools import wraps
from .logger import logger

def log_execution_time():
    def decorator(func):
        @wraps(func)
        def wrapper(*args,**kwargs):
            start = time.time()
            result = func(*args,**kwargs)
            elapsed = time.time() - start
            logger.info(f"{func.__name__} 耗时: {elapsed:.3f}秒")
            return result
        return wrapper
    return decorator

# # 使用示例
# @log_execution_time()
# def infer_img_wrapper(img):
#     return ppDocLayoutServer.infer_img(img)