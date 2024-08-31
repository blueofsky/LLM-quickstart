import torch
from transformers import AutoModelForCausalLM

model_name_or_path = 'THUDM/chatglm3-6b'   # 模型ID或本地路径
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="cuda").to(0)

for param in model.parameters():
    print(f'Parameter dtype: {param.dtype}')
    break 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
initial_memory_usage = torch.cuda.memory_allocated(device)
print(f"Initial memory usage: {initial_memory_usage/ (1024 ** 3)} GB")