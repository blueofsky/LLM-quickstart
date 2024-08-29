import torch
import gc
import evaluate
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from transformers import AutoModelForSpeechSeq2Seq, AutoTokenizer, AutoProcessor
from peft import PeftConfig, PeftModel
from datasets import load_dataset, DatasetDict, Audio
from torch.utils.data import DataLoader


model_name_or_path = "openai/whisper-large-v2"
model_dir = "models/whisper-large-v2-asr-int8"
language = "Chinese (China)"
language_abbr = "zh-CN"
task = "transcribe"
dataset_name = "mozilla-foundation/common_voice_11_0"
batch_size=16


peft_config = PeftConfig.from_pretrained(model_dir)
base_model = AutoModelForSpeechSeq2Seq.from_pretrained(
    peft_config.base_model_name_or_path, load_in_8bit=True, device_map="auto"
)
base_model.requires_grad_(False)
peft_model = PeftModel.from_pretrained(base_model, model_dir)
peft_model.eval()



tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path, language=language, task=task)
processor = AutoProcessor.from_pretrained(peft_config.base_model_name_or_path, language=language, task=task)
feature_extractor = processor.feature_extractor

common_voice = DatasetDict()
common_voice["test"] = load_dataset(dataset_name, language_abbr, split="test", trust_remote_code=True)
common_voice = common_voice.remove_columns(
    ["accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes"]
)
common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))
def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch

full_common_voice = DatasetDict()
full_common_voice["test"] = common_voice["test"]
tokenized_common_voice = full_common_voice.map(prepare_dataset)




# 定义一个针对语音到文本任务的数据整理器类
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any  # 处理器结合了特征提取器和分词器

    # 整理器函数，将特征列表处理成一个批次
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # 从特征列表中提取输入特征，并填充以使它们具有相同的形状
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # 从特征列表中提取标签特征（文本令牌），并进行填充
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # 使用-100替换标签中的填充区域，-100通常用于在损失计算中忽略填充令牌
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # 如果批次中的所有序列都以句子开始令牌开头，则移除它
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        # 将处理过的标签添加到批次中
        batch["labels"] = labels

        return batch  # 返回最终的批次，准备好进行训练或评估

# 用给定的处理器实例化数据整理器
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
# 词错误率（WER）是评估ASR模型常用的指标。从 Evaluate 加载 WER 指标
metric = evaluate.load("wer")
eval_dataloader = DataLoader(tokenized_common_voice["test"], batch_size=batch_size, collate_fn=data_collator)

# 遍历评估数据加载器中的所有批次
for step, batch in enumerate(tqdm(eval_dataloader)):
    # 使用自动混合精度来加速计算，并减少显存使用
    with torch.cuda.amp.autocast():
        # 不计算梯度，以节省计算资源，仅用于推理和评估
        with torch.no_grad():
            # 生成预测的标记(tokens)，这里使用模型的generate函数进行文本生成
            generated_tokens = (
                peft_model.generate(
                    input_features=batch["input_features"].to("cuda"),  # 将输入特征移动到GPU上
                    decoder_input_ids=batch["labels"][:, :4].to("cuda"),  # 提供解码器的初始输入
                    max_new_tokens=255,  # 设置生成的最大新标记数量
                )
                .cpu()  # 将生成的标记移回CPU
                .numpy()  # 转换为NumPy数组以便进一步处理
            )
            # 获取批次中的标签，并将其移回CPU
            labels = batch["labels"].cpu().numpy()
            # 将标签中的-100替换为填充标记的ID，-100通常用于忽略计算损失的标记
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            # 使用分词器解码生成的标记和标签，以获得可读的文本
            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            # 将预测和参考添加到评估指标中，用于后续的性能评估
            metric.add_batch(
                predictions=decoded_preds,
                references=decoded_labels,
            )
    # 删除不再需要的变量以释放内存
    del generated_tokens, labels, batch
    # 手动触发垃圾收集，进一步清理内存
    gc.collect()

# 计算词错误率（WER）指标，并将结果转换为百分比形式
wer = 100 * metric.compute()
# 打印词错误率，f"{wer=}"是一种格式化字符串的简洁写法，它会展示变量名和值
print(f"{wer=}%")