import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForQuestionAnswering,
    pipeline
)

# origin_model="distilbert-base-uncased"
finetune_model="models/distilbert-base-uncased-finetune-squad"
origin_model=finetune_model

def inference(question,context):
    tokenizer = AutoTokenizer.from_pretrained(origin_model)
    model = AutoModelForQuestionAnswering.from_pretrained(finetune_model)
    # 编码问题和上下文
    inputs = tokenizer.encode_plus(question, context, return_tensors="pt")

    # 推理
    with torch.no_grad():
        outputs = model(**inputs)
   
    # 提取答案
    answer_start_scores = outputs.start_logits
    answer_end_scores = outputs.end_logits
    answer_start = torch.argmax(answer_start_scores)
    answer_end = torch.argmax(answer_end_scores) + 1

    # 使用tokenizer解码答案
    answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end]))
    
    # print(answer_start,answer_start_scores,answer_end,answer_end_scores,sep="\n")

    # print("Score:", answer_start_scores[answer_start].item(), answer_end_scores[answer_end - 1].item())
    
    return answer,0
    


def pipd(question,context):
    pipe = pipeline(task="question-answering", tokenizer=origin_model, model=finetune_model)
    return pipe(question=question,context=context)

if __name__ == '__main__':    
    question = "What famous Elizabethan playwright spent much of his life in London?"
    context="""The pilgrims in Geoffrey Chaucer's late 14th-century Canterbury Tales set out for Canterbury from London – specifically, from the Tabard inn, Southwark. William Shakespeare spent a large part of his life living and working in London; his contemporary Ben Jonson was also based there, and some of his work—most notably his play The Alchemist—was set in the city. A Journal of the Plague Year (1722) by Daniel Defoe is a fictionalisation of the events of the 1665 Great Plague. Later important depictions of London from the 19th and early 20th centuries are Dickens' novels, and Arthur Conan Doyle's Sherlock Holmes stories. Modern writers pervasively influenced by the city include Peter Ackroyd, author of a "biography" of London, and Iain Sinclair, who writes in the genre of psychogeography."""
    predicted_label_idx,score=inference(question,context)
    print(f"Predicted label: 'answer': {predicted_label_idx}, 'score': {score:.4f}")
    predicted_label_idx=pipd(question=question, context=context)
    print(f"Predicted label: {predicted_label_idx}")