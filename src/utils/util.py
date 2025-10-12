import hashlib
import json
import os
import time
from pathlib import Path
import random

import tiktoken
import yaml
from langgraph.graph import StateGraph
from matplotlib import image as mpimg, pyplot as plt


def read_yaml_config(yaml_file: str) -> dict:
    with open(yaml_file, 'r') as f:
        return yaml.safe_load(f)

def set_api_key_environ(api_key_path: str) -> None:
    """
    Reads API keys and from the given path and sets the environ variable.
    """
    with open(api_key_path) as io:
        model_keys = json.load(io)

    for key, value in model_keys.items():
        os.environ[key] = value
        print(f"key: {key}: {value}")


def find_root_dir(start_path=None, marker='.git') -> None:
    """
    从当前目录向上查找包含指定标志文件的目录作为根目录。
    """
    if start_path is None:
        start_path = os.path.abspath(__file__)

    current_dir = os.path.dirname(start_path)
    while current_dir != os.path.dirname(current_dir):  # 到达文件系统根目录时停止
        if os.path.exists(os.path.join(current_dir, marker)):
            os.environ["ROOT_DIR"] = current_dir
            print("ROOT_DIR: ", current_dir)
            return
        current_dir = os.path.dirname(current_dir)

    raise FileNotFoundError(f"未找到包含 {marker} 的根目录")


# # 查找根目录
# root_dir = find_root_dir(marker='.git')
# print("根目录:", root_dir)


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
        mermaid_code = graph.get_graph(xray=1).draw_mermaid_png()
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

def calculate_token_usage(text, model="o200k_base") -> int:
    encoding = tiktoken.get_encoding(model)

    token = encoding.encode(text)

    return len(token)