from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM
)

quant_model_dir = "models/opt-6.7b-awq"
tokenizer = AutoTokenizer.from_pretrained(quant_model_dir)
model = AutoModelForCausalLM.from_pretrained(quant_model_dir, device_map="cuda").to(0)

# use quantized model for generation text
def generate_text(text):
    inputs = tokenizer(text, return_tensors="pt").to(0)
    out = model.generate(**inputs, max_new_tokens=64)
    return tokenizer.decode(out[0], skip_special_tokens=True)

result = generate_text("Merry Christmas! I'm glad to")
print(result)

result = generate_text("The woman worked as a")
print(result)