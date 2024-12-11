import hashlib
import json
import os
import time
from pathlib import Path
import random

import yaml
from langgraph.graph import StateGraph
from matplotlib import image as mpimg, pyplot as plt


def read_yaml_config(yaml_file: str) -> dict:
    with open(yaml_file, 'r') as f:
        return yaml.safe_load(f)

def set_api_key_environ(api_key_path: str) -> None:
    """
    Reads the API key from the given path and sets the environ variable.
    """
    with open(api_key_path) as io:
        keys = json.load(io)

    for key, value in keys.items():
        os.environ[key] = value
        print(os.environ[key])


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

def draw_lang_graph_flow(graph: StateGraph):
    try:
        mermaid_code = graph.get_graph().draw_mermaid_png()
        with open("graph.jpg", "wb") as f:
            f.write(mermaid_code)

        # 使用 matplotlib 显示图像
        img = mpimg.imread("graph.jpg")
        plt.imshow(img)
        plt.axis('off')  # 关闭坐标轴
        plt.show()

    except Exception as e:
        # This requires some extra dependencies and is optional
        print(e)
