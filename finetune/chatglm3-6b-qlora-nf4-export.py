from transformers import AutoModel, AutoTokenizer
from peft import PeftModel

model_name_or_path = 'THUDM/chatglm3-6b'
peft_model_path = "models/chatglm3-6b-nf4"
merge_model_path = "models/chatglm3-6b-nf4-merge"

# 加载基础模型
model = AutoModel.from_pretrained(model_name_or_path,trust_remote_code=True, device_map='auto')
# 加载基础模型的分词器
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

# 加载微调后的模型
peft_model = PeftModel.from_pretrained(model, peft_model_path)


# 模型合并
merge_model = peft_model.merge_and_unload()


# 保存合并后的模型
merge_model.save_pretrained(merge_model_path)