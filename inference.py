import argparse
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer
import os
import json
from peft import get_peft_model, LoraConfig, TaskType
from preprocess_utils import sanity_check, InputOutputDataset


# Argument Parser Setup
parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    type=str,
    default="chatglm3-6b-base",
    help="The directory of the model",
)
parser.add_argument(
    "--tokenizer",
    type=str,
    default="chatglm3-6b-base",
    help="Tokenizer path",
)
parser.add_argument("--LoRA", type=str, default=True, help="use lora or not")
parser.add_argument(
    "--lora-path",
    type=str,
    default="CEE_4epoch-20240128-135840-1e-4/checkpoint-20000/pytorch_model.pt",
    help="Path to the LoRA model checkpoint",
)
parser.add_argument(
    "--device", type=str, default="cuda", help="Device to use for computation"
)
parser.add_argument(
    "--max-new-tokens", type=int, default=128, help="Maximum new tokens for generation"
)
parser.add_argument("--lora-alpha", type=float, default=32, help="LoRA alpha")
parser.add_argument("--lora-rank", type=int, default=8, help="LoRA r")
parser.add_argument("--lora-dropout", type=float, default=0.1, help="LoRA dropout")

args = parser.parse_args()

if args.tokenizer is None:
    args.tokenizer = args.model

if args.LoRA:
    # Model and Tokenizer Configuration
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model, load_in_8bit=False, trust_remote_code=True, device_map="auto"
    ).to(args.device)

    # LoRA Model Configuration
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=True,
        target_modules=["query_key_value"],
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    model = get_peft_model(model, peft_config)
    if os.path.exists(args.lora_path):
        model.load_state_dict(torch.load(args.lora_path), strict=False)

else:
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model, load_in_8bit=False, trust_remote_code=True, device_map="auto"
    ).to(args.device)


# Interactive Prompt
# while True:
#     prompt = input("Prompt: ")
#     inputs = tokenizer(prompt, return_tensors="pt").to(args.device)
#     response = model.generate(input_ids=inputs["input_ids"],
#                               max_length=inputs["input_ids"].shape[-1] + args.max_new_tokens)
#     response = response[0, inputs["input_ids"].shape[-1]:]
#     print("Response:", tokenizer.decode(response, skip_special_tokens=True))

response_list = []
test_data_path = "test_meld_efr.json"

with open(test_data_path, "r", encoding="utf-8") as f:
    test_data = json.load(f)
for data in test_data:
    prompt = data["context"]
    inputs = tokenizer(prompt, return_tensors="pt").to(args.device)
    response = model.generate(
        input_ids=inputs["input_ids"],
        max_length=inputs["input_ids"].shape[-1] + args.max_new_tokens,
    )
    response = response[0, inputs["input_ids"].shape[-1] :]
    decode_response = tokenizer.decode(response, skip_special_tokens=True)
    response_list.append(decode_response)
    print("Response:", decode_response)

# Save the responses to a JSON file
save_path = "./newtrain.json"
with open(save_path, "w", encoding="utf-8") as f:
    json.dump(response_list, f, ensure_ascii=False, indent=4)

print(f"Responses saved to {save_path}")
