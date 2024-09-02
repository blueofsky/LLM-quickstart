import torch
from transformers import (
    AutoModel, 
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import (
    PeftModel,
    AutoPeftModelForCausalLM
)

def load_tokenizer(model_dir):
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, trust_remote_code=True
    )
    return tokenizer

def load_base_model(model_dir,q_config=None):
    if q_config:
        model = AutoModel.from_pretrained(
                    model_dir,
                    trust_remote_code=True, 
                    quantization_config=q_config,
                    device_map='auto'
                )
    else:
        model = AutoModel.from_pretrained(
                    model_dir,
                    trust_remote_code=True,
                    device_map='auto'
                )
    model.requires_grad_(False)
    model.eval()
    return model

def load_peft_model(model_dir,base_model):
    model = PeftModel.from_pretrained(base_model, model_dir)
    return model

def direct_load_peft_model(model_dir,q_config=None):
    if q_config:
        model = AutoPeftModelForCausalLM.from_pretrained(
                    model_dir, 
                    trust_remote_code=True, 
                    device_map='auto',                
                    quantization_config=q_config,
                )
    else:
        model = AutoPeftModelForCausalLM.from_pretrained(
                    model_dir, 
                    trust_remote_code=True, 
                    device_map='auto',
                )
    return model


if __name__ == "__main__":    
    directed=True # 是否直接加载FineTuning模型
    quanted=False   # 是否量化模型
    infered=False # 是否执行推理检查
    saved=True    # 是否保存最终模型
    
    # 加载分词器
    model_name_or_path = 'THUDM/chatglm3-6b'
    tokenizer = load_tokenizer(model_name_or_path)
    
    # 加载模型
    peft_model_path = "models/chatglm3-6b-nf4"
    q_config = BitsAndBytesConfig(load_in_4bit=True,
                                bnb_4bit_quant_type='nf4',
                                bnb_4bit_use_double_quant=True,
                                bnb_4bit_compute_dtype=torch.float32) if quanted else None
    base_model = load_base_model(model_name_or_path,q_config) if not directed else None
    if directed:
        peft_model = direct_load_peft_model(peft_model_path,q_config)
    else:
        peft_model = load_peft_model(peft_model_path,base_model) 
    
    # 合并模型
    merged_model = peft_model.merge_and_unload()
    
    # 模型推理
    if infered:
        input_text = '类型#上衣*材质#棉*颜色#白色*颜色#浅蓝色*风格#复古*图案#复古*衣样式#衬衫'
        base_response=''
        if base_model:
            base_response, _ = base_model.chat(tokenizer, input_text)
        ft_response,_ = peft_model.chat(tokenizer, input_text)
        mg_response,_= merged_model.chat(tokenizer, input_text)
        print(f"问题：{input_text}\n\n原始输出：\n{base_response}\n\n\n微调后：\n{ft_response}\n\n\n合并后：\n{mg_response}")
    
    # 保存模型
    if saved:        
        merge_model_path = "-".join([
            "/root/autodl-tmp/pretrains/chatglm3-6b-nf4",
            "direct" if directed else "",
            "qa" if quanted else "",
        ])
        merge_model_path.removesuffix('-')
        merged_model.save_pretrained(merge_model_path, safe_serialization=True)
        tokenizer.save_pretrained(merge_model_path)
    
