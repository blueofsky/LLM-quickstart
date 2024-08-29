from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

model_name_or_path = "facebook/opt-6.7b"
quant_model_dir = "models/opt-6.7b-gptq"

quantization_config = GPTQConfig(
     bits=4, # 量化精度
     group_size=128,
     dataset="wikitext2",
     desc_act=False,
)
# 模型量化
quant_model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    quantization_config=quantization_config,
    device_map='auto')


## 检查量化模型正确性
# 通过检查`线性层的属性`来确保模型已正确量化，它们应该包含`qweight`和`qzeros`属性，这些属性应该是`torch.int32`数据类型。
quant_model.model.decoder.layers[0].self_attn.q_proj.__dict__


# 保存模型权重
quant_model.save_pretrained(quant_model_dir)
# 保存分词器
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
tokenizer.save_pretrained(quant_model_dir)