from infer_online import run_inference
import torch

def get_best_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


if __name__ == "__main__":
    # 离线模式：把音频路径列表传进去，跑完就退出
    run_inference(
        checkpoint_dir="./checkpoints", 
        audio_paths=[     #These audios will be concated and send into model. 
            "assets/what_is_your_name.m4a",
            "assets/what_can_you_do.m4a",
        ],
        device=get_best_device(), 
    )