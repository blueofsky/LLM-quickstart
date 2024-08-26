"""
全量微调(full fine-tuning)

以下是代码的关键点，解释为什么它是全量微调：

1. **数据集加载与分词**：代码首先加载了一个数据集，并使用`AutoTokenizer`对数据集中的文本进行分词处理。

2. **模型加载**：使用`AutoModelForSequenceClassification.from_pretrained`加载了一个预训练的BERT模型，
    并指定了类别数（`num_labels=5`），这意味着模型的输出层（头）被调整为5个类别的分类任务。

3. **训练参数设置**：通过`TrainingArguments`设置了训练的参数，包括批次大小、epoch数等。

4. **Trainer初始化与训练**：初始化`Trainer`对象，并使用提供的训练和评估数据集、模型、训练参数和评估指标函数进行训练。

5. **模型评估与保存**：在训练完成后，使用测试数据集评估模型的性能，并将模型保存到指定目录。

全量微调的特点是使用整个预训练模型（包括所有层）进行训练，而不是仅仅训练模型的某些部分或最后的几层。
在这个例子中，模型从头到尾都在训练，没有迹象表明只有模型的某些部分（如最后几层）被微调。
此外，代码中没有使用任何特定的高效微调技术，如Adapters、Prompt Tuning或SFT。


"""

import numpy as np  # 用于数学运算
import evaluate  # 用于评估模型性能
from datasets import load_dataset  # 加载数据集
from transformers import (  # 导入Transformers库中的组件
    AutoTokenizer,  # 用于自动加载与模型匹配的分词器
    AutoModelForSequenceClassification,  # 用于自动加载与任务匹配的模型
    TrainingArguments,  # 用于设置训练参数
    Trainer  # 用于训练模型
)


# 数据预处理

# 加载数据集(650,000个训练样本和50,000个测试样本)
dataset = load_dataset("yelp_review_full")
# 加载与模型匹配的分词器
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
# 定义分词函数
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)  # 对文本进行分词，填充到最大长度并截断
# 使用分词函数处理数据集
tokenized_datasets = dataset.map(tokenize_function, batched=True)
# 从处理后的数据集中分离出训练集和测试集
train_dataset = tokenized_datasets["train"].shuffle(seed=64).select(range(10000))
eval_dataset = tokenized_datasets["test"].shuffle(seed=64).select(range(1000))



# 模型训练

# 定义模型保存的目录
model_dir = "models/bert-base-cased-finetune-yelp"
# 加载预训练模型并指定类别数
model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)
# 加载评估指标
metric = evaluate.load("accuracy")
# 定义计算评估指标的函数
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)  # 将模型输出的logits转换为预测类别
    return metric.compute(predictions=predictions, references=labels)  # 计算准确率
# 设置训练参数
training_args = TrainingArguments(
    output_dir=model_dir,  # 输出目录
    evaluation_strategy="epoch",  # 每个epoch后评估一次
    per_device_train_batch_size=64,  # 每个设备的训练批次大小
    per_device_eval_batch_size=64,  # 评估时每个设备上的批次大小
    num_train_epochs=3,  # 训练的epoch数
    logging_steps=100  # 日志记录的步数
)
# 初始化Trainer对象
trainer = Trainer(
    model=model,  # 模型
    args=training_args,  # 训练参数
    train_dataset=train_dataset,  # 训练数据集
    eval_dataset=eval_dataset,  # 评估数据集
    compute_metrics=compute_metrics  # 计算评估指标的函数
)
# 训练模型
trainer.train()


# 模型评估

# 从测试集中随机选择1000个样本进行评估
test_dataset = tokenized_datasets["test"].shuffle(seed=64).select(range(1000))
# 评估模型性能
trainer.evaluate(test_dataset)


# 模型保存

"""
保存模型到指定目录
作用：保存模型的当前状态，包括模型参数和配置。
用途：通常用于保存训练过程中的最佳模型或者最终训练完成的模型。这允许你在未来重新加载模型并进行进一步的评估或推理，而无需重新训练。
输出: 保存的文件通常包括模型的权重(pytorch_model.bin 或 tf_model.h5,
    取决于使用的是 PyTorch 还是 TensorFlow)和模型的配置文件(config.json)。
"""
trainer.save_model(model_dir)
"""
保存训练状态
作用: 保存整个训练状态,包括模型参数、优化器状态、训练轮次(epoch)、最佳评估指标等。
用途：用于保存训练过程中的完整状态，这样你可以在之后从这个状态继续训练，而不必从头开始。这在训练中断或需要暂停训练时非常有用。
输出：保存的是一个包含所有相关信息的字典，通常以 state_dict 的形式保存。
例子：
```python
    model = AutoModelForCausalLM.from_pretrained('path_to_saved_model')
    trainer = Trainer(model=model, args=training_args, ...)

    # 如果有保存的状态，可以使用 resume_from_checkpoint 参数
    trainer.train(resume_from_checkpoint='path_to_saved_state')
```
"""
trainer.save_state()