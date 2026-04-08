import json
import os
import pickle
import time

from openai import OpenAI
from tqdm import tqdm

client = OpenAI(
    api_key="your key",  # Replace with your key
    base_url="url" # Replace with your llm url
)
MODEL_NAME = "model_name" # Replace with your llm name, such as DeepSeek V3


def llm_analysis(content, model=MODEL_NAME):
    messages = [
        {
            "role": "system",
            "content": "You are a top-tier expert in social network analysis and information diffusion prediction. Please output the final result strictly in JSON format. Do not include any reasoning process, thinking steps, or extra explanatory text outside the JSON block."
        },
        {"role": "user", "content": content},
    ]
    start_time = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        # extra_body={
        #     "chat_template_kwargs": {"enable_thinking": False}  # For use with Qwen3.5 family models, enable CoT (set `enable_thinking` to `True`).
        # }
    )
    time_span = time.perf_counter() - start_time
    content = response.choices[0].message.content
    reasoning = None
    prompt_tokens = response.usage.prompt_tokens
    completion_tokens = response.usage.completion_tokens
    total_tokens = response.usage.total_tokens
    tokens = {'prompt': prompt_tokens, 'completion': completion_tokens, 'tokens': total_tokens}
    return time_span, content, reasoning, tokens


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pkl(path, content):
    with open(path, "wb") as f:
        pickle.dump(content, f)


def load_json(path):
    with open(path, "r", encoding='utf-8') as f:
        return json.load(f)


def save_json(path, content):
    with open(path, "w", encoding='utf-8') as f:
        f.write(json.dumps(content, ensure_ascii=False))


def main(save_path):
    input_content = '''
Your task is to complete the "Information Diffusion Prediction". I will provide you with a piece of [Information Content] to be propagated, along with the data of a set of [Candidate Users] (including their personal descriptions, social connections to the current information cascade, and historical behavior records).
Please comprehensively evaluate the probability of each candidate user forwarding, sharing, or participating in the dissemination of this information.'''

    output_content = '''
Please directly output the ranked list of user IDs (from highest to lowest propagation probability), followed by an overall predictive analysis summary explaining your core rationale for the ranking.

**Strict Formatting Requirement:** You must strictly output the result in JSON format (parsable by Python's json.loads). Do not include any Markdown tags (like ```json) or explanatory text outside the JSON. The structure must be exactly as follows:
{
  "ranked_user_ids": [user_id_A, user_id_B, user_id_C, ...],
  "summary": "Your overall predictive analysis summary explaining your core rationale for the ranking."
}'''

    save_dict = dict()
    if os.path.exists(save_path):
        temp_dict = load_json(save_path)
        for key, value in temp_dict.items():
            save_dict[int(key)] = value

    test_db = load_pkl('../datasets/test.pkl')
    news_dict = load_pkl('../datasets/news.pkl')
    users_dict = load_pkl('../datasets/users.pkl')

    for index, item in enumerate(tqdm(test_db[:515], desc='LLM')):
        if index in save_dict.keys():
            continue
        news_id, next_uid = item["news_id"], item["next_user"]

        # [Information Content]
        news_line = f"[Information Content]\nText: {news_dict[news_id]['text']}\n"

        candidate_users = item["neg_users"][:19]
        candidate_users.append(next_uid)
        history_uids = item["history_users"]

        # [Candidate Users Data]
        social_line = "[Candidate Users Data]\n"
        for uid in candidate_users:
            description = users_dict[uid]['description']
            social_line += f"User {uid}:\nProfile: The personal description of {uid} is {description}.\n"

            user_rels = [i for i in users_dict[uid]['social'] if i in history_uids]
            relation_count = len(user_rels)
            if relation_count > 0:
                social_line += f"Social Topology: {uid} has {relation_count} social connections with the current cascade, specifically: {' '.join(map(str, user_rels))}.\n"
            else:
                social_line += f"Social Topology: {uid} has 0 social connections with the current cascade.\n"

            history = users_dict[uid]['history'][:10]
            history = [str(i)[:10].replace('\n', '. ') for i in history]
            if len(history) > 0:
                social_line += f"Historical Behavior: The historical behaviors of {uid} are: {' | '.join(map(str, history))}.\n"
            else:
                social_line += f"Historical Behavior: The historical behaviors of {uid} are: Not provided.\n"

            social_line += '-' * 10 + '\n'

        prompt = f'{input_content}\n{news_line}\n{social_line}\n{output_content}'
        time_span, content, reasoning, tokens = llm_analysis(content=prompt)

        save_dict[index] = {'time_span': time_span, 'content': content, 'reasoning': reasoning, 'tokens': tokens}
        save_json(save_path, save_dict)


if __name__ == '__main__':
    save_path = f'../saves/{MODEL_NAME}.json'
    main(save_path)
