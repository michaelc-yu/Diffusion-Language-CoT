import openai  # or other API client
import tqdm

def evaluate_with_llm(examples, config):
    model = config.get("judge_model", "gpt-4")

    scored = []
    for ex in tqdm(examples, desc="LLM judging"):
        prompt = (
            f"Question:\n{ex['input']}\n\n"
            f"Reference Answer:\n{ex['target']}\n\n"
            f"Model Output:\n{ex['prediction']}\n\n"
            "Is the model's answer correct? Respond with 'Correct' or 'Incorrect' and a short explanation."
        )
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        judgment = response['choices'][0]['message']['content']

        ex["llm_judgment"] = judgment
        ex["is_correct"] = "correct" in judgment.lower()
        scored.append(ex)

    return scored
