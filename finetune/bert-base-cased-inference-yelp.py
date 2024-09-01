import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    pipeline
)

origin_model="bert-base-cased"
finetune_model="models/bert-base-cased-finetune-yelp"

def inference(text):
    # 模型加载
    tokenizer = AutoTokenizer.from_pretrained(origin_model)
    model = AutoModelForSequenceClassification.from_pretrained(finetune_model)
    
    #使用分词器将输入文本转换为模型所需的格式，并指定返回的张量类型为PyTorch张量（"pt"代表PyTorch）
    #这里的inputs变量将包含分词器处理后的结果，它是一个字典，包含分词器生成的编码张量、注意力掩码（attention mask）等信息
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    
    # 模型推理
    with torch.no_grad():
        outputs = model(**inputs)
    ##获取了模型的输出logits。Logits是模型在每个可能的类别上的得分或概率，还没有经过概率归一化处理
    logits = outputs.logits
    
    # 归一化
    ## 使用PyTorch中的torch.softmax函数对logits进行softmax操作。softmax函数将每个logit值转换为在[0, 1]范围内的概率值，使得所有类别的概率之和为1
    ## dim=-1表示在最后一个维度上进行softmax操作。在情感分类任务中，通常最后一个维度是类别维度
    probabilities = torch.softmax(logits, dim=-1)
    #使用argmax找到logits中概率最高的类别索引，并通过.item()将其转换为一个普通的Python整数。
    predicted_label_idx = torch.argmax(probabilities, dim=1).item()
    
    #使用预测得到的类别索引predicted_label_idx从id2label映射中获取对应的标签。
    print('id2label: ',model.config.id2label)
    predicted_label_name = model.config.id2label.get(predicted_label_idx)
    
    # score
    score = probabilities[0].tolist()[predicted_label_idx]
    return predicted_label_name,score

def pipd(text):
    pipe = pipeline(task="sentiment-analysis", tokenizer=origin_model, model=finetune_model, device=0)
    print('pipeline use: ',pipe.model.device)
    return pipe(text)

if __name__ == '__main__':    
    text = "One of the best pizzas I've had in Charlotte. They also sell pizza by the slice which is a plus"
    predicted_label_idx,score=inference(text)
    print(f"Predicted label: 'label': {predicted_label_idx}, 'score': {score:.4f}")
    predicted_label_idx=pipd(text)
    print(f"Predicted label: {predicted_label_idx}")