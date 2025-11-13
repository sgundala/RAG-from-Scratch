%%time

# Tokenize the input text (turn it into numbers) and send it to GPU
input_ids = tokenizer(prompt, return_tensors="pt").to("cuda")
print(f"Model input (tokenized):\n{input_ids}\n")

# Generate outputs based on the tokenized input
outputs = llm_model.generate(
    **input_ids,
    max_new_tokens=256   # define the maximum number of new tokens to create
)

print(f"Model output (tokens):\n{outputs[0]}\n")
