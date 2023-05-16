from transformers import pipeline, set_seed

prompts_gen_model = pipeline('text-generation', model='gpt2')


def generate_prompts(model, starting_phrase, max_length, num_return_sequences):
    set_seed(42)
    prompts = model(
        starting_phrase, max_length=max_length,
        num_return_sequences=num_return_sequences
    )
    return [prompt["generated_text"] for prompt in prompts]


generated_prompts = generate_prompts(prompts_gen_model, "draw a picture of ", 10, 1)
for prompt in generated_prompts:
    print(prompt)