import json
import pickle

import numpy as np


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_json(path):
    with open(path, "r", encoding='utf-8') as f:
        return json.load(f)


def evaluate_metrics(target_ids, pred_lists, ks=[1, 2, 5]):
    results = {f'HITS@{k}': [] for k in ks}
    results.update({f'MAP@{k}': [] for k in ks})
    results.update({f'NDCG@{k}': [] for k in ks})

    for target, pred in zip(target_ids, pred_lists):
        try:
            rank = pred.index(target) + 1
        except ValueError:
            rank = float('inf')

        for k in ks:
            if rank <= k:
                results[f'HITS@{k}'].append(1)
                results[f'MAP@{k}'].append(1.0 / rank)
                results[f'NDCG@{k}'].append(1.0 / np.log2(rank + 1))
            else:
                results[f'HITS@{k}'].append(0)
                results[f'MAP@{k}'].append(0)
                results[f'NDCG@{k}'].append(0)

    final_metrics = {metric: np.mean(values) for metric, values in results.items()}
    return final_metrics


test_li = load_pkl(path='../datasets/test.pkl')
pred_dict = load_json(path='../saves/DiffAgent.json')
target_ids, pred_lists = [], []
for index, item in pred_dict.items():
    test_item = test_li[int(index)]
    next_user = test_item['next_user']
    target_ids.append(next_user)
    content = json.loads(item['content'])
    rank_users = content['ranked_user_ids']
    pred_lists.append(rank_users)

metrics = evaluate_metrics(target_ids=target_ids, pred_lists=pred_lists)
for k, v in metrics.items():
    print(k, '\t', round(v, 4))
