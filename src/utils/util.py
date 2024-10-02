import hashlib
import json
import os
import time
from pathlib import Path
import random


def set_api_key_environ(api_key_path: str) -> None:
    """
    Reads the API key from the given path and sets the environ variable.
    """
    with open(api_key_path) as io:
        keys = json.load(io)

    for key, value in keys.items():
        os.environ[key] = value
        # print(os.environ[key])


def generate_md5_id() -> str:
    timestamp = str(time.time())

    # 生成一个随机数
    random_number = str(random.randint(0, 100000))

    # 将时间戳和随机数拼接起来
    unique_string = timestamp + random_number

    # 生成 MD5 哈希
    md5_hash = hashlib.md5(unique_string.encode())

    # 返回 MD5 哈希的十六进制表示
    return md5_hash.hexdigest()
