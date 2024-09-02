import torch
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig

# 加载微调模型的路径
peft_model_path ="models/chatglm3-6b-nf4"
# 从预训练好的微调模型中加载配置
peft_config = PeftConfig.from_pretrained(peft_model_path)
# 配置量化设置，用于模型的4比特量化
q_config = BitsAndBytesConfig(load_in_4bit=True,
                              bnb_4bit_quant_type='nf4',
                              bnb_4bit_use_double_quant=True,
                              bnb_4bit_compute_dtype=torch.float32)
# 根据微调配置加载基础模型，并应用量化配置
base_model = AutoModel.from_pretrained(peft_config.base_model_name_or_path,
                                       quantization_config=q_config,
                                       trust_remote_code=True,
                                       device_map='auto')
# 冻结基础模型参数，不进行梯度更新
base_model.requires_grad_(False)
# 将基础模型设置为评估模式
base_model.eval()

# 加载微调后的模型
peft_model = PeftModel.from_pretrained(base_model, peft_model_path)
# 加载基础模型的分词器
tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path, trust_remote_code=True)


# 推理函数，用于比较原始模型和微调后模型的输出
def compare_results(query, base_model, peft_model):
    """
    比较并输出原始模型和微调后模型的推理结果。
    
    参数:
    - query: 用户的查询或问题
    - base_model: 原始基础模型
    - peft_model: 微调后的模型
    
    无返回值，直接打印比较结果。
    """
    # 获取原始模型的响应
    base_response, _ = base_model.chat(tokenizer, query)
    # 获取微调后模型的响应
    ft_response,_ = peft_model.chat(tokenizer, query)
    # 打印比较结果
    print(f"问题：{query}\n\n原始输出：\n{base_response}\n\n\n微调后：\n{ft_response}")

# 输入文本，包含多个关键词
input_text = '类型#裙*版型#显瘦*风格#文艺*风格#简约*图案#印花*图案#撞色*裙下摆#压褶*裙长#连衣裙*裙领型#圆领'
# 调用函数，比较并输出结果
compare_results(input_text, base_model, peft_model)