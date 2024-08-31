import torch
from transformers import AutoModel,AutoTokenizer

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
def print_cuda_memory_gb(model,tokenizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    initial_memory_usage = torch.cuda.memory_allocated(device)
    vocab_size = tokenizer.vocab_size
    print(f"Initial memory usage: {initial_memory_usage/ (1024 ** 3)} GB,Vocab size: {vocab_size}")
    input_ids = torch.randint(0, vocab_size, (1, 512)).to(device)
    with torch.no_grad():
        model(input_ids)
    # 计算显存占用
    final_memory_usage = torch.cuda.memory_allocated(device)
    print(f"Final memory usage: {final_memory_usage} bytes")
    # 计算差异
    memory_used = final_memory_usage - initial_memory_usage
    print(f"Memory used by the model: {memory_used} bytes")


if __name__ == "__main__":
    model_name_or_path = 'THUDM/chatglm3-6b'   # 模型ID或本地路径
    model = AutoModel.from_pretrained(model_name_or_path,trust_remote_code=True, device_map="cuda").to(0)    
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,trust_remote_code=True)
    print_model_dtype(model)
    print_memory_gb(model)
    print_cuda_memory_gb(model,tokenizer)
    print_memory_gb(model)
    
