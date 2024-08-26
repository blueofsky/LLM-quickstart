import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    pipeline
)

origin_model="bert-base-cased"
finetune_model="models/bert-base-cased-finetune-yelp"

def inference(text):
    tokenizer = AutoTokenizer.from_pretrained(origin_model)
    model = AutoModelForSequenceClassification.from_pretrained(finetune_model)
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=1)
    predicted_label_idx = torch.argmax(probabilities, dim=1).item()
    class_names = ["LABEL_0", "LABEL_1", "LABEL_2", "LABEL_3", "LABEL_4"]
    predicted_label_name = class_names[predicted_label_idx]
    return predicted_label_name

def pipd(text):
    pipe = pipeline(task="sentiment-analysis", tokenizer=origin_model, model=finetune_model)
    return pipe(text)

if __name__ == '__main__':    
    text = "One of the best pizzas I've had in Charlotte. They also sell pizza by the slice which is a plus"
    predicted_label_idx=inference(text)
    print(f"Predicted label name: {predicted_label_idx}")
    predicted_label_idx=pipd(text)
    print(f"Predicted label: {predicted_label_idx}")