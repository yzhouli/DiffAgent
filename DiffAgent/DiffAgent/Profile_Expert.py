import json
import os
import pickle
import time
from openai import OpenAI

client = OpenAI(
    api_key="your key",  # Replace with your key
    base_url="url"  # Replace with your llm url
)
MODEL_NAME = "model_name"  # Replace with your llm name, such as DeepSeek V3

def build_text_messages(system_prompt, content):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content}
    ]
    return messages


def llm_analysis(system_prompt, content, model=MODEL_NAME):
    messages = build_text_messages(system_prompt, content)
    start_time = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        extra_body={
            "chat_template_kwargs": {"enable_thinking": False}
        }
    )
    time_span = time.perf_counter() - start_time
    content = response.choices[0].message.content

    tokens = {
        'prompt': response.usage.prompt_tokens,
        'completion': response.usage.completion_tokens,
        'tokens': response.usage.total_tokens
    }
    return time_span, content, None, tokens


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


class ProfileExpert:
    def __init__(self, users_pkl_path='../datasets/users.pkl'):
        print("Initializing Profile Expert and loading user datasets...")
        self.users_dict = load_pkl(users_pkl_path)

        self.system_prompt = (
            "You are the Profile Expert in the DiffAgent architecture. "
            "Your objective is to evaluate the individual susceptibility of candidate users "
            "towards a specific topic."
        )

    def evaluate_susceptibility(self, parsed_semantics, candidate_uids, instruction=None):
        user_contexts_str = ""
        for uid in candidate_uids:
            if uid in self.users_dict:
                description = self.users_dict[uid].get('description', 'No description available')

                history_raw = self.users_dict[uid].get('history', [])[:10]
                history_clean = [str(i)[:10].replace('\n', '. ') for i in history_raw]
                history_str = ' | '.join(history_clean) if history_clean else "Not provided"

                user_contexts_str += f"User {uid}: Static Profile <{description}>, Historical Behavior <{history_str}>.\n"
            else:
                user_contexts_str += f"User {uid}: Data not found in system.\n"

        output_req = instruction if instruction else (
            "For each candidate, directly output a brief paragraph evaluating their individual susceptibility "
            "based on the semantic resonance between their historical behaviors and the topic."
        )

        user_prompt = f"""
The system has automatically loaded the parsed topic semantics and the user context data for your evaluation:
Task Directive: Evaluate how susceptible the following candidates are to the target topic based on their personal historical alignment.
Parsed Topic Semantics:
{parsed_semantics}

Target Candidate IDs: {candidate_uids}

System-Loaded User Contexts:
{user_contexts_str}
[OUTPUT REQUIREMENT]
{output_req}"""

        time_span, content, reasoning, tokens = llm_analysis(
            system_prompt=self.system_prompt,
            content=user_prompt
        )

        return {
            "candidate_uids": candidate_uids,
            "content": content,
            "time_span": time_span,
            "tokens": tokens,
            "used_instruction": output_req
        }