import torch
from transformers import AutoModelForCausalLM

# 获取模型dtype
def print_model_dtype(model):
    for param in model.parameters():
        print(f'Parameter dtype: {param.dtype}')
        break 

# 获取模型占用的 GPU显存（差值为预留给 PyTorch 的显存）
def print_memory_gb(model):
    memory_footprint_bytes = model.get_memory_footprint()
    memory_footprint_mib = memory_footprint_bytes / (1024 ** 3)  # 转换为 GB
    print(f"Model Memory usage: {memory_footprint_mib:.2f}GB")
    
# 获取GPU显存占用
def print_cuda_memory_gb():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    initial_memory_usage = torch.cuda.memory_allocated(device)
    print(f"Cuda memory usage: {initial_memory_usage/ (1024 ** 3)} GB")


if __name__ == "__main__":
    model_name_or_path = 'THUDM/chatglm3-6b'   # 模型ID或本地路径
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="cuda").to(0)
    print_model_dtype(model)
    print_memory_gb(model)
    print_cuda_memory_gb()
    
