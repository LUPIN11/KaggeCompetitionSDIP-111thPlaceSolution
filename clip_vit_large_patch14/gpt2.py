import pandas as pd
from tqdm import tqdm
from transformers import pipeline

model = pipeline('text-generation', model='./gpt2')


def generate_prompts(core="try imagine this imaginative picture: ", num_prompts=10):
    return model(
        core,
        max_length=60,
        num_return_sequences=num_prompts,
        temperature=0.70,
        eos_token_id=model.tokenizer.convert_tokens_to_ids("."),
        early_stopping=True,
        top_k=800,
        top_p=800,
        pad_token_id=model.tokenizer.convert_tokens_to_ids(" "),
    )


candidates = []
# count = 0
print('Begin Generating')
for _ in tqdm(range(10000), ncols=100):
    candidates.extend([sen['generated_text'] for sen in generate_prompts()])
df = pd.DataFrame({'prompt': candidates})
df.to_csv('new_prompts_1m.csv')
