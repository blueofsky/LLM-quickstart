import torch
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

model_id = "facebook/opt-6.7b"
# 获取模型占用的 GPU显存（差值为预留给 PyTorch 的显存）
def print_memory_mib(model):
    memory_footprint_bytes = model.get_memory_footprint()
    memory_footprint_mib = memory_footprint_bytes / (1024 ** 2)  # 转换为 MiB
    print(f"{memory_footprint_mib:.2f}MiB")
    

    
### 使用 Transformers 库的 `model.from_pretrained()`方法中的`load_in_8bit`或`load_in_4bit`参数，便可以对模型进行量化。
# 只要模型支持使用Accelerate加载并包含torch.nn.Linear层，这几乎适用于任何模态的任何模型。
model_4bit = AutoModelForCausalLM.from_pretrained(model_id,device_map="auto",load_in_4bit=True)
print_memory_mib(model_4bit)


### 使用 NF4 精度加载模型
nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
)
model_nf4 = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=nf4_config)
print_memory_mib(model_nf4)


### 使用双量化加载模型
double_quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
)
model_double_quant = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=double_quant_config)
print_memory_mib(model_double_quant)


### 使用 QLoRA 所有量化技术加载模型
qlora_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)
model_qlora = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=qlora_config)
print_memory_mib(model_qlora)