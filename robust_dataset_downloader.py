#!/usr/bin/env python3
"""
改进版 BEIR 数据集下载脚本 - 带重试和断点续传
"""
import os
import sys
import json
import time
import shutil
import hashlib
from pathlib import Path
from typing import Optional, Tuple
import urllib.request
import urllib.error
from zipfile import ZipFile
from beir.datasets.data_loader import GenericDataLoader

class RobustDownloader:
    """带重试机制的稳健下载器"""
    
    def __init__(self, max_retries=5, retry_delay=10):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
    
    def download_with_retry(self, url: str, dest_path: str) -> bool:
        """下载文件，支持重试和断点续传"""
        print(f"下载 URL: {url}")
        print(f"目标路径: {dest_path}")
        
        # 创建目标目录
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        
        for attempt in range(1, self.max_retries + 1):
            try:
                print(f"\n尝试 {attempt}/{self.max_retries}...")
                
                # 检查是否有部分下载的文件
                resume_pos = 0
                if os.path.exists(dest_path):
                    resume_pos = os.path.getsize(dest_path)
                    print(f"发现部分下载的文件，从 {resume_pos} 字节继续...")
                
                # 创建请求
                req = urllib.request.Request(url)
                if resume_pos > 0:
                    req.add_header('Range', f'bytes={resume_pos}-')
                
                # 下载文件
                mode = 'ab' if resume_pos > 0 else 'wb'
                with urllib.request.urlopen(req, timeout=60) as response:
                    total_size = int(response.headers.get('Content-Length', 0))
                    if resume_pos > 0:
                        total_size += resume_pos
                    
                    print(f"总大小: {total_size / (1024*1024):.2f} MB")
                    
                    with open(dest_path, mode) as f:
                        downloaded = resume_pos
                        chunk_size = 8192
                        last_print = time.time()
                        
                        while True:
                            chunk = response.read(chunk_size)
                            if not chunk:
                                break
                            
                            f.write(chunk)
                            downloaded += len(chunk)
                            
                            # 每秒打印一次进度
                            if time.time() - last_print >= 1:
                                if total_size > 0:
                                    percent = (downloaded / total_size) * 100
                                    print(f"进度: {downloaded / (1024*1024):.2f} MB / "
                                          f"{total_size / (1024*1024):.2f} MB ({percent:.1f}%)",
                                          end='\r')
                                else:
                                    print(f"已下载: {downloaded / (1024*1024):.2f} MB", end='\r')
                                last_print = time.time()
                
                print(f"\n✓ 下载完成！")
                return True
                
            except (urllib.error.URLError, ConnectionResetError, 
                    TimeoutError, OSError) as e:
                print(f"\n✗ 下载失败: {e}")
                
                if attempt < self.max_retries:
                    wait_time = self.retry_delay * attempt
                    print(f"等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                else:
                    print(f"已达到最大重试次数，放弃下载")
                    return False
            
            except KeyboardInterrupt:
                print("\n用户中断下载")
                return False
        
        return False
    
    def extract_zip(self, zip_path: str, extract_to: str) -> bool:
        """解压 ZIP 文件"""
        try:
            print(f"\n解压文件: {zip_path}")
            print(f"解压到: {extract_to}")
            
            with ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
            
            print("✓ 解压完成")
            return True
            
        except Exception as e:
            print(f"✗ 解压失败: {e}")
            return False


def process_dataset(dataset_name: str, data_dir: str, 
                   skip_if_exists: bool = True) -> bool:
    """下载并处理单个数据集"""
    print(f"\n{'='*70}")
    print(f"处理数据集: {dataset_name}")
    print(f"{'='*70}\n")
    
    # 创建数据集目录
    dataset_path = Path(data_dir) / dataset_name
    dataset_path.mkdir(parents=True, exist_ok=True)
    
    # 检查输出文件是否已存在
    output_file = dataset_path / f"{dataset_name}_documents.jsonl"
    if skip_if_exists and output_file.exists():
        line_count = sum(1 for _ in open(output_file))
        print(f"✓ 文件已存在，跳过下载: {output_file}")
        print(f"  文档数量: {line_count}")
        return True
    
    # 下载 ZIP 文件
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
    zip_path = dataset_path / f"{dataset_name}.zip"
    
    downloader = RobustDownloader(max_retries=5, retry_delay=10)
    
    # 如果 ZIP 文件不完整或不存在，重新下载
    if not zip_path.exists() or not is_valid_zip(zip_path):
        if zip_path.exists():
            print(f"ZIP 文件不完整，重新下载...")
            zip_path.unlink()
        
        if not downloader.download_with_retry(url, str(zip_path)):
            print(f"✗ 下载失败: {dataset_name}")
            return False
    else:
        print(f"✓ ZIP 文件已存在: {zip_path}")
    
    # 解压文件
    extract_dir = dataset_path / "raw"
    if not extract_dir.exists():
        if not downloader.extract_zip(str(zip_path), str(dataset_path)):
            print(f"✗ 解压失败: {dataset_name}")
            return False
    else:
        print(f"✓ 数据已解压")
    
    # 查找解压后的数据目录
    data_folders = list(dataset_path.glob("*"))
    data_folder = None
    for folder in data_folders:
        if folder.is_dir() and folder.name != "raw":
            if (folder / "corpus.jsonl").exists() or (folder / "collection.tsv").exists():
                data_folder = folder
                break
    
    if not data_folder:
        print(f"✗ 找不到数据文件: {dataset_name}")
        return False
    
    print(f"数据目录: {data_folder}")
    
    # 加载数据
    print(f"\n加载数据...")
    try:
        corpus, queries, qrels = GenericDataLoader(str(data_folder)).load(split="test")
    except Exception as e:
        print(f"✗ 加载数据失败: {e}")
        return False
    
    # 转换为 JSONL 格式
    print(f"转换为 JSONL 格式: {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for doc_id, doc_data in corpus.items():
            title = doc_data.get('title', '').strip()
            text = doc_data.get('text', '').strip()
            doctext = (title + ' ' + text).strip() if title else text
            
            doc_entry = {
                'docid': doc_id,
                'doctext': doctext
            }
            f.write(json.dumps(doc_entry, ensure_ascii=False) + '\n')
    
    print(f"\n✓ 完成 {dataset_name}")
    print(f"  - 文档数量: {len(corpus)}")
    print(f"  - 查询数量: {len(queries)}")
    print(f"  - 相关性判断数量: {len(qrels)}")
    print(f"  - 输出文件: {output_file}")
    
    # 可选：删除 ZIP 文件以节省空间
    # zip_path.unlink()
    
    return True


def is_valid_zip(zip_path: Path) -> bool:
    """检查 ZIP 文件是否完整"""
    try:
        with ZipFile(zip_path, 'r') as zip_ref:
            # 尝试读取文件列表
            zip_ref.namelist()
        return True
    except:
        return False


def main():
    if len(sys.argv) < 3:
        print("用法: python robust_dataset_downloader.py <dataset_name> <data_dir> [--force]")
        print("\n可用选项:")
        print("  --force    强制重新下载，即使文件已存在")
        print("\n示例:")
        print("  python robust_dataset_downloader.py fiqa ~/projects/moe-q/data")
        sys.exit(1)
    
    dataset_name = sys.argv[1]
    data_dir = sys.argv[2]
    force_download = "--force" in sys.argv
    
    print(f"开始处理数据集: {dataset_name}")
    print(f"数据目录: {data_dir}")
    print(f"强制下载: {force_download}")
    
    success = process_dataset(
        dataset_name, 
        data_dir, 
        skip_if_exists=not force_download
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
