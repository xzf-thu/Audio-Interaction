from huggingface_hub import snapshot_download
import os

repo_id = "zhifeixie/StreamAudio-2M"
local_dir = "./checkpoints/StreamAudio-2M"

if os.path.exists(local_dir):
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
    )

    print(f"Downloaded {repo_id} to {local_dir}")

else:
    print("You are not in the right position. cd ./Mini-Omni3 and use 'export PYTHONPATH=./'")