import json
import os
import pickle
import time
import base64
import cv2
from PIL import Image
from io import BytesIO
import numpy as np
from openai import OpenAI

client = OpenAI(
    api_key="key",  # Replace with your key
    base_url="url"  # Replace with your llm url
)
MODEL_NAME = "llm name"  # Replace with your llm name, such as DeepSeek V3


def encode_image(image_path, max_size=(800, 600)):
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


def extract_video_frames(video_path, num_frames=8, max_size=800):
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


def build_messages(system_prompt, content, mm_path):
    messages = [{"role": "system", "content": system_prompt}]

    if 'None' == mm_path or not os.path.exists(mm_path):
        messages.append({"role": "user", "content": content})
        return messages

    user_content = []
    if '.mp4' in mm_path.lower():
        frames = extract_video_frames(mm_path, num_frames=8)
        for frame_b64 in frames:
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"}
            })
    else:
        img_b64 = encode_image(mm_path)
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
        })

    user_content.append({"type": "text", "text": content})
    messages.append({"role": "user", "content": user_content})

    return messages


def llm_analysis(system_prompt, content, mm_path, model=MODEL_NAME):
    messages = build_messages(system_prompt, content, mm_path)
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
    reasoning = None
    prompt_tokens = response.usage.prompt_tokens
    completion_tokens = response.usage.completion_tokens
    total_tokens = response.usage.total_tokens
    tokens = {'prompt': prompt_tokens, 'completion': completion_tokens, 'tokens': total_tokens}
    return time_span, content, reasoning, tokens


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


class SemanticExpert:
    def __init__(self, news_pkl_path='../datasets/news.pkl', mm_dir_path='../datasets/mm'):
        print("Initializing Semantic Expert and loading datasets...")
        self.news_dict = load_pkl(news_pkl_path)
        self.mm_dir_path = mm_dir_path
        self.cache = dict()

        # [SYSTEM INSTRUCTION]
        self.system_prompt = (
            "You are the Semantic Expert in the DiffAgent architecture. "
            "Your objective is to decipher the underlying diffusion motivations, "
            "core semantics, and emotional tendencies of a given topic."
        )

    def gat_cache(self, news_id):
        if news_id in self.cache.keys():
            return self.cache[news_id]
        return None

    def analyze_topic(self, news_id, instruction=None):
        if news_id not in self.news_dict:
            return {"error": f"Topic ID {news_id} not found in the dataset."}
        cache_out = self.gat_cache(news_id=news_id)
        if cache_out is not None:
            return cache_out

        news_data = self.news_dict[news_id]
        mm_path = os.path.join(self.mm_dir_path, str(news_data['mm_path']))
        text_content = news_data['text']

        output_req = instruction if instruction else (
            "Directly output a single paragraph summarizing the topic and "
            "analyzing its underlying diffusion motivations."
        )

        user_prompt = f"""[USER INPUT]
The Coordinator Agent has assigned you a Topic_ID. To assist you, the underlying system has automatically executed the load topic module and injected the corresponding raw data below.
Target Topic ID: {news_id}
System-Loaded Textual Content: {text_content[:1000]}
System-Loaded Multimodal Content: [See attached Image/Video]

[OUTPUT REQUIREMENT]
{output_req}"""

        time_span, content, reasoning, tokens = llm_analysis(
            system_prompt=self.system_prompt,
            content=user_prompt,
            mm_path=mm_path
        )

        out = {
            "news_id": news_id,
            "content": content,
            "time_span": time_span,
            "tokens": tokens,
            "used_instruction": output_req
        }

        self.cache[news_id] = out

        return out
