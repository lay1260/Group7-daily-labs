import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="bert-base-chinese",
    local_dir="./local_bert_base",
    local_dir_use_symlinks=False,
    resume_download=True,
    # 只下载PyTorch需要的文件，排除其他框架的冗余文件
    ignore_patterns=["*.h5", "*.msgpack", "*.pb", "*.ckpt"]
)