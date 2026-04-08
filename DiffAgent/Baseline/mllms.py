import json
import os
import pickle
import time
import base64
import cv2
import numpy as np
from PIL import Image
from openai import OpenAI
from io import BytesIO
from tqdm import tqdm

client = OpenAI(
    api_key="your key",  # Replace with your key
    base_url="url"  # Replace with your llm url
)
MODEL_NAME = "llm name"  # Replace with your llm name, such as DeepSeek V3


def encode_image(image_path, max_size=(400, 300)):
    with Image.open(image_path) as img:
        if img.mode in ('RGBA', 'LA', 'P'):
            background = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'P':
                img = img.convert('RGBA')
            background.paste(img, mask=img.split()[3] if img.mode == 'RGBA' else None)
            img = background
        elif img.mode != 'RGB':
            img = img.convert('RGB')

        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        buffered = BytesIO()
        img.save(buffered, format='JPEG', quality=85, optimize=True)
        return base64.b64encode(buffered.getvalue()).decode('utf-8')


def extract_video_frames(video_path, num_frames=8, max_size=400):
    cap = cv2.VideoCapture(video_path)
    frames_b64 = []
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return frames_b64
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        return frames_b64
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            h, w = frame.shape[:2]
            if max(w, h) > max_size:
                scale = max_size / max(w, h)
                new_w, new_h = int(w * scale), int(h * scale)
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            frame_b64 = base64.b64encode(buffer).decode('utf-8')
            frames_b64.append(frame_b64)
    cap.release()
    return frames_b64


def build_messages(content, mm_path):
    system_prompt = "You are a top-tier expert in social network analysis and information diffusion prediction. Please output the final result strictly in JSON format. Do not include any reasoning process, thinking steps, or extra explanatory text outside the JSON block."

    messages = [
        {
            "role": "system",
            "content": system_prompt
        }
    ]

    if 'None' == mm_path:
        messages.append({
            "role": "user",
            "content": content
        })
        return messages

    user_content = []

    if '.mp4' in mm_path.lower():
        frames = extract_video_frames(mm_path, num_frames=8)
        for frame_b64 in frames:
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{frame_b64}"
                }
            })
    else:
        img_b64 = encode_image(mm_path)
        user_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{img_b64}"
            }
        })

    user_content.append({
        "type": "text",
        "text": content
    })

    messages.append({
        "role": "user",
        "content": user_content
    })

    return messages


def llm_analysis(content, mm_path, model=MODEL_NAME):
    messages = build_messages(content, mm_path)
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
    mm_dir_path = '../datasets/mm'

    for index, item in enumerate(tqdm(test_db[:515], desc='LLM')):
        if index in save_dict.keys():
            continue
        news_id, next_uid = item["news_id"], item["next_user"]

        # [Information Content]
        mm_path = mm_dir_path + '/' + news_dict[news_id]['mm_path']
        news_line = f"[Information Content]\nText: {news_dict[news_id]['text']}\nMultimodal: Understand multimodal content and comprehensively grasp the topic\n"

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
        time_span, content, reasoning, tokens = llm_analysis(content=prompt, mm_path=mm_path)

        save_dict[index] = {'time_span': time_span, 'content': content, 'reasoning': reasoning, 'tokens': tokens}
        save_json(save_path, save_dict)


if __name__ == '__main__':
    save_path = f'../saves/{MODEL_NAME}.json'
    main(save_path)
