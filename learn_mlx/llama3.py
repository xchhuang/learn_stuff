from mlx_lm import load, generate


# model, tokenizer = load("mlx-community/Meta-Llama-3-8B-8bit")
# response = generate(model, tokenizer, prompt="do you know how to play Go", verbose=True)


model, tokenizer = load("mlx-community/Meta-Llama-3-8B-Instruct")
response = generate(model, tokenizer, prompt="hello", verbose=True)
