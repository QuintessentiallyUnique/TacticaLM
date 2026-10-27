import os
from huggingface_hub import snapshot_download

print("Starting robust download via huggingface_hub...")
snapshot_download(repo_id="Qwen/Qwen3-4B-Instruct-2507", ignore_patterns=["*.pt", "*.pth", "*.msgpack"])
print("Download complete!")
