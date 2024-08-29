import torch
from transformers import (
    AutomaticSpeechRecognitionPipeline,
    AutoModelForSpeechSeq2Seq,
    AutoTokenizer,
    AutoProcessor
)
from peft import (
    PeftConfig,
    PeftModel
)

model_dir = "models/whisper-large-v2-asr-int8"
language = "Chinese (China)"
language_abbr = "zh-CN"
language_decode = "chinese"
task = "transcribe"

# 
peft_config = PeftConfig.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path, language=language, task=task)
processor = AutoProcessor.from_pretrained(peft_config.base_model_name_or_path, language=language, task=task)
feature_extractor = processor.feature_extractor
# forced_decoder_ids = processor.get_decoder_prompt_ids(language=language_decode, task=task)
base_model = AutoModelForSpeechSeq2Seq.from_pretrained(
    peft_config.base_model_name_or_path, 
    load_in_8bit=True, 
    device_map="auto"
)
peft_model = PeftModel.from_pretrained(base_model, model_dir)

# 
test_audio = "../peft/data/audio/test_zh.flac"
pipeline = AutomaticSpeechRecognitionPipeline(
    model=peft_model,
    tokenizer=tokenizer, 
    feature_extractor=feature_extractor
)
with torch.cuda.amp.autocast():
    text = pipeline(test_audio, max_new_tokens=255)["text"]
print(text)