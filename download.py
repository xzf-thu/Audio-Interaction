from huggingface_hub import snapshot_download
import os

repo_id = "zhifeixie/AudioInteraction-2M"  # 改成你的模型 repo_id
local_dir = "./checkpoints/"

if os.path.exists(local_dir):
    snapshot_download(
        repo_id=repo_id,
        repo_type="model",   # 下载模型
        local_dir=local_dir,
        resume_download=True,
    )

    print(f"Downloaded model {repo_id} to {local_dir}")

else:
    print("You are not in the right position. cd ./AudioInteraction and use 'export PYTHONPATH=./'")