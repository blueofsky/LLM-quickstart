import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    pipeline
)

origin_model="distilbert-base-uncased"
finetune_model="models/distilbert-base-uncased-finetune-squad"

def inference(text):
    tokenizer = AutoTokenizer.from_pretrained(origin_model)
    model = AutoModelForSequenceClassification.from_pretrained(finetune_model)
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=1)
    predicted_label_idx = torch.argmax(probabilities, dim=1).item()
    return predicted_label_idx

def pipd(text):
    pipe = pipeline(task="question-answering", tokenizer=origin_model, model=finetune_model)
    return pipe(text)

if __name__ == '__main__':    
    text = "What famous Elizabethan playwright spent much of his life in London?"
    # predicted_label_idx,score=inference(text)
    # print(f"Predicted label: 'label': {predicted_label_idx}, 'score': {score:.4f}")
    predicted_label_idx=pipd(text)
    print(f"Predicted label: {predicted_label_idx}")