import pickle
import tiktoken

with open('lichess_data.pkl', 'rb') as f:
    data = pickle.load(f)

tokenizer = tiktoken.get_encoding('o200k_base')

total_tokens = 0

for pos, move in data:
    text = f'Board: {pos}\nMove: {move}'
    tokens = len(tokenizer.encode(text))
    total_tokens += tokens

print(total_tokens)
