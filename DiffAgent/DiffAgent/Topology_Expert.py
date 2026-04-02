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



class TopologyExpert:
    def __init__(self, users_pkl_path='../datasets/users.pkl'):
        print("Initializing Topology Expert and loading user datasets...")
        self.users_dict = load_pkl(users_pkl_path)

        self.system_prompt = (
            "You are the Topology Expert in the DiffAgent architecture. "
            "Your objective is to quantify the structural social attraction exerted on "
            "candidate users by the current information cascade."
        )

    def evaluate_topology(self, cascade_uids, candidate_uids, instruction=None):
        topology_contexts_str = ""
        for uid in candidate_uids:
            if uid in self.users_dict:
                user_social_network = self.users_dict[uid].get('social', [])
                intersecting_connections = [str(i) for i in user_social_network if i in cascade_uids]
                connection_density = len(intersecting_connections)

                if connection_density > 0:
                    details = f"{connection_density} connections ({', '.join(intersecting_connections)})"
                else:
                    details = "0 connections"

                topology_contexts_str += f"User {uid}: Heterogeneous social relations intersecting with the active cascade <{details}>.\n"
            else:
                topology_contexts_str += f"User {uid}: Topology data not found in system.\n"

        output_req = instruction if instruction else (
            "For each candidate, directly output a brief paragraph evaluating their structural "
            "social attraction based on the density and specific connections they have with the currently infected users."
        )

        user_prompt = f"""[USER INPUT]
The system has automatically loaded the active cascade sequence and the relevant sub-graph connection data for your evaluation:
Task Directive: Assess the structural pull on the following candidates originating from the currently infected users.

Current Cascade Sequence: {cascade_uids}

Target Candidate IDs: {candidate_uids}

System-Loaded Social Topology:
{topology_contexts_str}
[OUTPUT REQUIREMENT]
{output_req}"""

        time_span, content, reasoning, tokens = llm_analysis(
            system_prompt=self.system_prompt,
            content=user_prompt
        )

        return {
            "cascade_uids": cascade_uids,
            "candidate_uids": candidate_uids,
            "content": content,
            "time_span": time_span,
            "tokens": tokens,
            "used_instruction": output_req
        }