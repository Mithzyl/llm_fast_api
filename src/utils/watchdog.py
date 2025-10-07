import time
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, DirDeletedEvent, FileDeletedEvent, FileSystemEvent
import os

# --- 配置日志，方便观察事件 ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

class PersonalDocumentHandler(FileSystemEventHandler):
    """
    自定义事件处理器，专注于处理新创建的文件。
    """
    def on_created(self, event: FileSystemEvent):
        # 我们只关心文件的创建事件，忽略目录的创建
        if not event.is_directory:
            file_path = event.src_path
            logging.info(f"检测到新文件: {file_path}")
            
            # --- 这是关键的集成点 ---
            # 在这里，我们将调用处理文档的函数
            # 例如: process_and_embed_document(file_path)
            print(f"准备处理文件: {file_path}")

    def on_deleted(self, event: DirDeletedEvent | FileDeletedEvent) -> None:
        print("检测到文件删除, 删除相应内容分块")


