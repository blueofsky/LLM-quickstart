import torch
from typing import List, Dict
from datasets import load_dataset
from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING
from peft import (
    TaskType, 
    LoraConfig, 
    get_peft_model, 
    prepare_model_for_kbit_training
)
from transformers import (
    AutoTokenizer,
    AutoModel, 
    BitsAndBytesConfig,
    TrainingArguments, 
    Trainer
)

# 定义全局变量和参数
peft_model_path = "models/chatglm3-6b-nf4" # 训练好的模型存储目录
peft_data_path="data/adgen"
model_name_or_path = 'THUDM/chatglm3-6b'   # 模型ID或本地路径
model_revision='b098244'                   # 模型版本
train_data_path = 'HasturOfficial/adgen'   # 训练数据路径

## 数据处理
dataset = load_dataset(train_data_path)
# revision='b098244' 版本对应的 ChatGLM3-6B 设置 use_reentrant=False
# 最新版本 use_reentrant 被设置为 True，会增加不必要的显存开销
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,trust_remote_code=True,revision=model_revision)
max_input_length = 512                    # 输入的最大长度
max_output_length = 1536                  # 输出的最大长度
prompt_text = ''                          # 所有数据前的指令文本
# tokenize_func 函数
def tokenize_func(example, tokenizer, ignore_label_id=-100):
    """
    对单个数据样本进行tokenize处理。

    参数:
    example (dict): 包含'content'和'summary'键的字典，代表训练数据的一个样本。
    tokenizer (transformers.PreTrainedTokenizer): 用于tokenize文本的tokenizer。
    ignore_label_id (int, optional): 在label中用于填充的忽略ID，默认为-100。

    返回:
    dict: 包含'tokenized_input_ids'和'labels'的字典，用于模型训练。
    """

    # 构建问题文本
    question = prompt_text + example['content']
    if example.get('input', None) and example['input'].strip():
        question += f'\n{example["input"]}'

    # 构建答案文本
    answer = example['summary']

    # 对问题和答案文本进行tokenize处理
    q_ids = tokenizer.encode(text=question, add_special_tokens=False)
    a_ids = tokenizer.encode(text=answer, add_special_tokens=False)

    # 如果tokenize后的长度超过最大长度限制，则进行截断
    if len(q_ids) > max_input_length - 2:  # 保留空间给gmask和bos标记
        q_ids = q_ids[:max_input_length - 2]
    if len(a_ids) > max_output_length - 1:  # 保留空间给eos标记
        a_ids = a_ids[:max_output_length - 1]

    # 构建模型的输入格式
    input_ids = tokenizer.build_inputs_with_special_tokens(q_ids, a_ids)
    question_length = len(q_ids) + 2  # 加上gmask和bos标记

    # 构建标签，对于问题部分的输入使用ignore_label_id进行填充
    labels = [ignore_label_id] * question_length + input_ids[question_length:]

    return {'input_ids': input_ids, 'labels': labels}
# 
tokenized_dataset = dataset['train'].shuffle(seed=8).select(range(1000))
print(tokenized_dataset[:5])
tokenized_dataset = tokenized_dataset.map(
    lambda example: tokenize_func(example, tokenizer),
    batched=False, 
    remove_columns=tokenized_dataset.column_names
).flatten_indices()
# 保存数据集
# tokenized_dataset.save_to_disk(peft_data_path)



## 数据整理
class DataCollatorForChatGLM:
    """
    用于处理批量数据的DataCollator，尤其是在使用 ChatGLM 模型时。

    该类负责将多个数据样本（tokenized input）合并为一个批量，并在必要时进行填充(padding)。

    属性:
    pad_token_id (int): 用于填充(padding)的token ID。
    max_length (int): 单个批量数据的最大长度限制。
    ignore_label_id (int): 在标签中用于填充的ID。
    """

    def __init__(self, pad_token_id: int, max_length: int = 2048, ignore_label_id: int = -100):
        """
        初始化DataCollator。

        参数:
        pad_token_id (int): 用于填充(padding)的token ID。
        max_length (int): 单个批量数据的最大长度限制。
        ignore_label_id (int): 在标签中用于填充的ID，默认为-100。
        """
        self.pad_token_id = pad_token_id
        self.ignore_label_id = ignore_label_id
        self.max_length = max_length

    def __call__(self, batch_data: List[Dict[str, List]]) -> Dict[str, torch.Tensor]:
        """
        处理批量数据。

        参数:
        batch_data (List[Dict[str, List]]): 包含多个样本的字典列表。

        返回:
        Dict[str, torch.Tensor]: 包含处理后的批量数据的字典。
        """
        # 计算批量中每个样本的长度
        len_list = [len(d['input_ids']) for d in batch_data]
        batch_max_len = max(len_list)  # 找到最长的样本长度

        input_ids, labels = [], []
        for len_of_d, d in sorted(zip(len_list, batch_data), key=lambda x: -x[0]):
            pad_len = batch_max_len - len_of_d  # 计算需要填充的长度
            # 添加填充，并确保数据长度不超过最大长度限制
            ids = d['input_ids'] + [self.pad_token_id] * pad_len
            label = d['labels'] + [self.ignore_label_id] * pad_len
            if batch_max_len > self.max_length:
                ids = ids[:self.max_length]
                label = label[:self.max_length]
            input_ids.append(torch.LongTensor(ids))
            labels.append(torch.LongTensor(label))

        # 将处理后的数据堆叠成一个tensor
        input_ids = torch.stack(input_ids)
        labels = torch.stack(labels)

        return {'input_ids': input_ids, 'labels': labels}
data_collator = DataCollatorForChatGLM(pad_token_id=tokenizer.pad_token_id)


## 量化配置
# 使用 `nf4` 量化数据类型加载模型，开启双量化配置，以`bf16`混合精度训练，预估显存占用接近4GB
q_config = BitsAndBytesConfig(load_in_4bit=True,
                              bnb_4bit_quant_type='nf4',
                              bnb_4bit_use_double_quant=True,
                              bnb_4bit_compute_dtype=torch.bfloat16 # 计算数据类型（fp32, fp16, bf16）
                            )
# revision='b098244' 版本对应的 ChatGLM3-6B 设置 use_reentrant=False
# 最新版本 use_reentrant 被设置为 True，会增加不必要的显存开销
model = AutoModel.from_pretrained(model_name_or_path,
                                  quantization_config=q_config,
                                  device_map='auto',
                                  trust_remote_code=True,
                                  revision=model_revision)
# 获取模型占用的 GPU显存（差值为预留给 PyTorch 的显存）
def print_memory_mib(model):
    memory_footprint_bytes = model.get_memory_footprint()
    memory_footprint_mib = memory_footprint_bytes / (1024 ** 2)  # 转换为 MiB
    print(f"{memory_footprint_mib:.2f}MiB")

print_memory_mib(model)



## LoRA配置                                
kbit_model = prepare_model_for_kbit_training(model) # 预处理量化后的模型，使其可以支持低精度微调训练
target_modules = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING['chatglm']
lora_config = LoraConfig(
    target_modules=target_modules,
    r=4,
    lora_alpha=32,
    lora_dropout=0.05,
    bias='none',
    inference_mode=False,
    task_type=TaskType.CAUSAL_LM
)
qlora_model = get_peft_model(kbit_model, lora_config)
qlora_model.print_trainable_parameters()


## 模型训练
train_epochs=3     # 训练轮数
training_args = TrainingArguments(
    output_dir=peft_model_path,          # 输出目录
    per_device_train_batch_size=48,                     # 每个设备的训练批量大小
    gradient_accumulation_steps=4,                     # 梯度累积步数
    # per_device_eval_batch_size=8,                      # 每个设备的评估批量大小
    learning_rate=1e-3,                                # 学习率
    num_train_epochs=train_epochs,                     # 训练轮数
    lr_scheduler_type="linear",                        # 学习率调度器类型
    warmup_ratio=0.1,                                  # 预热比例
    logging_steps=10,                                  # 日志记录步数
    save_strategy="steps",                             # 模型保存策略
    save_steps=100,                                    # 模型保存步数
    # evaluation_strategy="steps",                      # 评估策略
    # eval_steps=500,                                   # 评估步数
    optim="adamw_torch",                               # 优化器类型
    fp16=True,                                         # 是否使用混合精度训练
)
trainer = Trainer(
        model=qlora_model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator
)
trainer.train()


## 保存模型
#   用途：通常用于保存带有前缀微调（Prefix Tuning）或参数高效微调（Parameter-Efficient Fine-Tuning, PEFT）的模型。
#   保存内容：不仅保存模型权重，还会保存模型配置文件和其他相关文件（如tokenizer配置等）。
#   适用场景：适用于使用了特定微调技术的情况，如LoRA、Adapter等。
trainer.model.save_pretrained(peft_model_path)
