from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "Aqal-1.0-1.5B-instruct-lora",  # YOUR MODEL YOU USED FOR TRAINING
    max_seq_length = 1048,
    dtype = None,
    load_in_4bit = False,
)
FastLanguageModel.for_inference(model)  # Enable native 2x faster inference

# alpaca_prompt = You MUST copy from above!
alpaca_prompt_urdu = """ذیل میں ایک ہدایت ہے جو ایک کام کی وضاحت کرتی ہے۔ ایک جواب لکھیں جو مناسب طریقے سے درخواست کو مکمل کرے۔ 
#### ہدایات: 
# {} 

#### جواب: 
# {}"""

inputs = tokenizer(
    [
        alpaca_prompt_urdu.format(
            # "Describe the planet Earth extensively.", # instruction
            "سیارے زمین کی وسیع پیمانے پر وضاحت کریں۔",
            "",  # output - leave this blank for generation!
        ),
    ],
    return_tensors = "pt",
).to("cuda")


from transformers import TextStreamer

text_streamer = TextStreamer(tokenizer)
_ = model.generate(
    **inputs, streamer = text_streamer, max_new_tokens = 128, repetition_penalty = 0.1
)