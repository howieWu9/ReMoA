import os
import sys
import json
from beir import util
from beir.datasets.data_loader import GenericDataLoader

def process_dataset(dataset_name, data_dir):
    """下载并处理单个数据集"""
    print(f"\n{'='*60}")
    print(f"处理数据集: {dataset_name}")
    print(f"{'='*60}\n")
    
    # 创建数据集目录
    dataset_path = os.path.join(data_dir, dataset_name)
    os.makedirs(dataset_path, exist_ok=True)
    
    # 下载数据集
    print(f"下载 {dataset_name} 数据集...")
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
    try:
        downloaded_path = util.download_and_unzip(url, dataset_path)
    except Exception as e:
        print(f"下载失败: {e}")
        return False
    
    # 加载数据
    print(f"加载数据...")
    try:
        corpus, queries, qrels = GenericDataLoader(downloaded_path).load(split="test")
    except Exception as e:
        print(f"加载数据失败: {e}")
        return False
    
    # 转换为 DUQGen 格式
    output_file = os.path.join(dataset_path, f"{dataset_name}_documents.jsonl")
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
    
    print(f"✓ 完成 {dataset_name}")
    print(f"  - 文档数量: {len(corpus)}")
    print(f"  - 查询数量: {len(queries)}")
    print(f"  - 输出文件: {output_file}")
    
    return True

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python process_dataset.py <dataset_name> <data_dir>")
        sys.exit(1)
    
    dataset_name = sys.argv[1]
    data_dir = sys.argv[2]
    
    success = process_dataset(dataset_name, data_dir)
    sys.exit(0 if success else 1)
