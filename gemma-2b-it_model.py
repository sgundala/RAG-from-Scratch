import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import is_flash_attn_2_available
from transformers import BitsAndBytesConfig

# 1. Create quantization config for smaller model loading (optional)
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
)

# Bonus: Setup Flash Attention 2 for faster inference,
# default to "sdpa" if it's not available
if is_flash_attn_2_available() and torch.cuda.get_device_capability(0)[0] >= 8:
    attn_implementation = "flash_attention_2"
else:
    attn_implementation = "sdpa"

print(f"[INFO] Using attention implementation: {attn_implementation}")

# 2. Pick a model id we’d like to use
model_id = "google/gemma-2b-it"
print(f"[INFO] Using model_id: {model_id}")

# 3. Instantiate tokenizer
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_id)

# 4. Instantiate the model
llm_model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_id,
    torch_dtype=torch.float16,
    quantization_config=quantization_config,
    low_cpu_mem_usage=False,
    attn_implementation=attn_implementation,
)
