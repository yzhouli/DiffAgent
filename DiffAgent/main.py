import json
import os
import pickle
import time
from tqdm import tqdm

from DiffAgent.Coordinator_Agent import CoordinatorAgent
from DiffAgent.Profile_Expert import ProfileExpert
from DiffAgent.Semantic_Expert import SemanticExpert
from DiffAgent.Topology_Expert import TopologyExpert

MODEL_NAME = "DiffAgent"

def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_json(path):
    with open(path, "r", encoding='utf-8') as f:
        return json.load(f)


def save_json(path, content):
    with open(path, "w", encoding='utf-8') as f:
        f.write(json.dumps(content, ensure_ascii=False))

def main(save_path):
    save_dict = dict()
    if os.path.exists(save_path):
        temp_dict = load_json(save_path)
        for key, value in temp_dict.items():
            save_dict[int(key)] = value

    print(f"Loaded {len(save_dict)} existing records from {save_path}")

    test_db = load_pkl('datasets/test.pkl')

    print("\nInitializing DiffAgent System...")
    semantic_expert = SemanticExpert(news_pkl_path='datasets/news.pkl', mm_dir_path='datasets/mm')
    profile_expert = ProfileExpert(users_pkl_path='datasets/users.pkl')
    topology_expert = TopologyExpert(users_pkl_path='datasets/users.pkl')

    coordinator = CoordinatorAgent(
        semantic_expert=semantic_expert,
        profile_expert=profile_expert,
        topology_expert=topology_expert,
        max_steps=5
    )
    print("System Initialization Complete.\n")

    pdbr = tqdm(total=len(test_db))
    for index, item in enumerate(test_db[:515]):
        pdbr.desc = 'DiffAgent Eval'
        pdbr.update(1)
        if index in save_dict.keys():
            continue

        news_id = item["news_id"]
        next_uid = item["next_user"]

        candidate_users = item["neg_users"][:19]
        candidate_users.append(next_uid)

        history_uids = item["history_users"]

        start_time = time.perf_counter()
        final_decision, memory = coordinator.predict_diffusion(
            topic_id=news_id,
            candidate_uids=candidate_users,
            cascade_uids=history_uids
        )
        content_str = json.dumps(final_decision, ensure_ascii=False)
        time_span = time.perf_counter() - start_time

        save_dict[index] = {
            'time_span': time_span,
            'content': content_str,
            'memory': memory,
            'reasoning': "Multi-Agent ReAct Loop Executed",
            'tokens': None
        }

        save_json(save_path, save_dict)


if __name__ == '__main__':
    save_path = f'saves/{MODEL_NAME}.json'
    main(save_path)
