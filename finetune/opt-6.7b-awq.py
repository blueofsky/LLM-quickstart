from awq import AutoAWQForCausalLM
from transformers import (
    AutoTokenizer,
    AwqConfig
)

## 加载模型
def load_model_and_tokenizer(model_name_or_path):
    try:
        model = AutoAWQForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=False)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=False)
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        return None, None
# 量化模型
def quantize_model(model, tokenizer, quant_config):
    if not (0 < quant_config["w_bit"] <= 8 and 0 < quant_config["q_group_size"]):
        raise ValueError("Invalid quantization configuration.")
    
    # 修改配置文件以使其与transformers集成兼容
    quantization_settings = AwqConfig(
        bits=quant_config["w_bit"],
        group_size=quant_config["q_group_size"],
        zero_point=quant_config["zero_point"],
        version=quant_config["version"].lower(),
    ).to_dict()
    # 预训练的transformers模型存储在model属性中，我们需要传递一个字典
    model.model.config.quantization_config = quantization_settings
    # 模型量化
    model.quantize(tokenizer, quant_config=quant_config)
    return model

# 保存模型权重
def save_model_and_tokenizer(model, tokenizer, quant_model_dir):
    model.save_quantized(quant_model_dir)
    tokenizer.save_pretrained(quant_model_dir)

# 执行
model_name_or_path = "facebook/opt-6.7b"
quant_model_dir = "models/opt-6.7b-awq"
quant_config = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM"
}
model, tokenizer = load_model_and_tokenizer(model_name_or_path)
if model is not None and tokenizer is not None:
    model = quantize_model(model, tokenizer, quant_config)
    save_model_and_tokenizer(model, tokenizer, quant_model_dir)
    
