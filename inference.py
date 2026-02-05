from unsloth import FastLanguageModel
from transformers import TextIteratorStreamer
from threading import Thread
import textwrap
max_print_width = 100
# Before running inference, call `FastLanguageModel.for_inference` first
from models.model_loader import load_model

model_name = "llama3_cpt_tinystories_final"
model, tokenizer = load_model(model_name)
print(f"Loaded model: {model}")


# Check if model has LoRA / PEFT adapters
if hasattr(model, "is_peft") and model.is_peft:
    print("Model has LoRA adapters attached.")
else:
    print("Model is plain base model (no LoRA adapters).")

FastLanguageModel.for_inference(model)

text_streamer = TextIteratorStreamer(tokenizer)
inputs = tokenizer(
[
    "Once upon a time"
]*1, return_tensors = "pt").to("cuda")

generation_kwargs = dict(
    inputs,
    streamer = text_streamer,
    max_new_tokens = 256,
    use_cache = True,
)
thread = Thread(target = model.generate, kwargs = generation_kwargs)
thread.start()

length = 0
for j, new_text in enumerate(text_streamer):
    if j == 0:
        wrapped_text = textwrap.wrap(new_text, width = max_print_width)
        length = len(wrapped_text[-1])
        wrapped_text = "\n".join(wrapped_text)
        print(wrapped_text, end = "")
    else:
        length += len(new_text)
        if length >= max_print_width:
            length = 0
            print()
        print(new_text, end = "")
    pass
pass

