import json
import re
import time
from openai import OpenAI

client = OpenAI(
    api_key="sk-g7KYh5jEVzuzPOw7E755Eb810bAd44F79aB482E85414207a",  # Replace with your key
    base_url="https://aihubmix.com/v1"  # Replace with your llm url
)
MODEL_NAME = "qwen3.5-35b-a3b"  # Replace with your llm name, such as DeepSeek V3


def build_text_messages(system_prompt, content):
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content}
    ]


def llm_analysis(system_prompt, content, model=MODEL_NAME, enable_thinking=True):
    messages = build_text_messages(system_prompt, content)
    start_time = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        extra_body={
            "chat_template_kwargs": {"enable_thinking": enable_thinking}
        }
    )
    time_span = time.perf_counter() - start_time
    return time_span, response.choices[0].message.content


class ACMemoryAgent:
    def __init__(self):
        print("Initializing AC Memory Agent...")

    def update_memory(self, previous_memory, new_observation):
        # 如果是第一步，记忆为空，直接返回新观察结果作为初始记忆
        if not previous_memory or previous_memory.strip() == "Memory is currently empty.":
            return new_observation

        print("   -> [Memory Agent] Running Actor-Critic memory integration...")

        # 1. Actor Phase: Draft the combined memory
        actor_sys = """You are the Memory Actor. 
Your task is to integrate the [New Observation] into the [Previous Memory].
Condense the information, resolve any contradictions (favoring the newer observation if it corrects a previous assumption), and output a concise, updated memory draft.
Maintain key data points like Topic IDs, User IDs, and specific evaluation metrics."""

        actor_user = f"[Previous Memory]:\n{previous_memory}\n\n[New Observation]:\n{new_observation}\n\nDraft the updated memory:"
        _, drafted_memory = llm_analysis(actor_sys, actor_user, enable_thinking=False)

        # 2. Critic Phase: Review and Validate
        critic_sys = """You are the Memory Critic.
Review the [Drafted Memory] against the original [Previous Memory] and [New Observation].
1. Did the Actor drop any critical User IDs or numerical scores?
2. Is the logical flow coherent?
Fix any omissions or hallucinations. Output ONLY the finalized, verified memory content without any conversational filler or meta-commentary."""

        critic_user = f"[Previous Memory]:\n{previous_memory}\n\n[New Observation]:\n{new_observation}\n\n[Drafted Memory from Actor]:\n{drafted_memory}\n\nProvide the final validated memory content:"
        _, final_memory = llm_analysis(critic_sys, critic_user, enable_thinking=False)

        return final_memory.strip()


class CoordinatorAgent:
    def __init__(self, semantic_expert, profile_expert, topology_expert, max_steps=5):
        print("Initializing Coordinator Agent...")
        self.semantic_expert = semantic_expert
        self.profile_expert = profile_expert
        self.topology_expert = topology_expert
        self.max_steps = max_steps
        self.memory_agent = ACMemoryAgent()

        self.system_prompt = """You are the Coordinator Agent within the DiffAgent architecture.
        Your primary responsibility is to orchestrate an asynchronous cognitive reasoning loop to predict information diffusion paths in social networks.

        Initial State Constraints:
        You are only provided with the target [Topic_ID] and a list of [Candidate_User_IDs].

        Action Space:
        Action 1: Call Semantic Expert to parse multimodal diffusion motivations.
        Action 2: Call Profile Expert to evaluate individual susceptibility.
        Action 3: Call Topology Expert to quantify structural social attraction.
        Action 4: Final Decision to terminate the loop when evidence is sufficient.

        Output Formatting:
        For EVERY iteration, regardless of which Action you choose (1, 2, 3, or 4), you MUST output your cognitive state in the following format. 
        Your `Action_Output` MUST ALWAYS be a valid JSON containing any required tool parameters AND your current running prediction:

        Thought: [Analyze the current Memory and identify missing evidence or update your logic.]
        Action: [Select ONE action: 'Action 1', 'Action 2', 'Action 3', or 'Action 4']
        Action_Output: {
            "topic_id": "<ID, only if Action 1>",
            "user_ids": ["<IDs, only if Action 2 or 3>"],
            "ranked_user_ids": [<Your CURRENT BEST GUESS of sorted candidate user IDs>],
            "summary": "<Your transparent logical explanation based on current memory>"
        }"""

    def parse_llm_response(self, text):
        thought = re.search(r'Thought:\s*(.*?)(?=Action:|$)', text, re.DOTALL | re.IGNORECASE)
        action = re.search(r'Action:\s*(.*?)(?=Action_Output:|$)', text, re.DOTALL | re.IGNORECASE)
        action_output = re.search(r'Action_Output:\s*(.*?)$', text, re.DOTALL | re.IGNORECASE)

        return {
            "Thought": thought.group(1).strip() if thought else "",
            "Action": action.group(1).strip() if action else "",
            "Action_Output": action_output.group(1).strip() if action_output else "{}"
        }

    def extract_json(self, text):
        try:
            start = text.find('{')
            end = text.rfind('}') + 1
            if start != -1 and end != 0:
                return json.loads(text[start:end])
            return {}
        except Exception as e:
            print(f"JSON Parsing Error: {e}")
            return {}

    def predict_diffusion(self, topic_id, candidate_uids, cascade_uids):
        global_working_memory = "Memory is currently empty."
        step = 1
        parsed_topic_semantics = ""
        latest_decision = {}

        step_records = []

        print(f"\n[Coordinator Agent] Starting Prediction for Topic {topic_id}")

        while step <= self.max_steps:
            print(f"\n--- Step {step}/{self.max_steps} ---")

            user_prompt = f"""[USER INPUT]
Current Step: {step} / {self.max_steps}
Target Topic ID: {topic_id}
Candidate User IDs: {candidate_uids}

Global Working Memory:
{global_working_memory}

[OUTPUT REQUIREMENT]
Based on the current state, generate your next Thought, Action, and Action_Output.
CRITICAL: Your Action_Output MUST be a valid JSON containing "ranked_user_ids" and "summary" reflecting your current understanding, even if you are choosing Action 1, 2, or 3."""

            _, llm_response = llm_analysis(self.system_prompt, user_prompt)

            parsed = self.parse_llm_response(llm_response)
            action_type = parsed["Action"]
            action_params = self.extract_json(parsed["Action_Output"])

            print(f"Thought: {parsed['Thought'][:100]}...")
            print(f"Action Executed: {action_type}")

            if action_params and "ranked_user_ids" in action_params:
                latest_decision = action_params

            if "Action 1" in action_type:
                target_topic = action_params.get("topic_id", topic_id)
                print(f"-> System implicitly calling Semantic Expert for Topic {target_topic}...")
                result = self.semantic_expert.analyze_topic(target_topic)
                parsed_topic_semantics = result.get("content", "Error fetching semantic data.")
                current_observation = f"[Observation Step {step} - Semantic Data]: {parsed_topic_semantics}"

            elif "Action 2" in action_type:
                target_users = action_params.get("user_ids", candidate_uids)
                print(f"-> System implicitly calling Profile Expert for Users {target_users}...")
                if not parsed_topic_semantics:
                    observation = "[System Error]: Cannot run Profile Expert without Parsed Topic Semantics. Call Action 1 first."
                else:
                    result = self.profile_expert.evaluate_susceptibility(parsed_topic_semantics, target_users)
                    observation = result.get("content", "Error fetching profile data.")
                current_observation = f"[Observation Step {step} - Susceptibility Profiles]: {observation}"

            elif "Action 3" in action_type:
                target_users = action_params.get("user_ids", candidate_uids)
                print(f"-> System implicitly calling Topology Expert for Users {target_users}...")
                result = self.topology_expert.evaluate_topology(cascade_uids, target_users)
                observation = result.get("content", "Error fetching topology data.")
                current_observation = f"[Observation Step {step} - Topological Attraction]: {observation}"

            elif "Action 4" in action_type:
                print("-> Final Decision Reached!")

                step_records.append({
                    "step": step,
                    "action": action_type,
                    "decision": action_params,
                    "memory_after_step": global_working_memory
                })

                return action_params, step_records

            else:
                current_observation = f"[System Error]: Invalid action '{action_type}'. Please select Action 1, 2, 3, or 4."

            if current_observation:
                global_working_memory = self.memory_agent.update_memory(
                    previous_memory=global_working_memory,
                    new_observation=current_observation
                )
                print(f"-> Memory Updated via AC Loop. Current Memory Length: {len(global_working_memory)} chars.")

            step_records.append({
                "step": step,
                "action": action_type,
                "decision": latest_decision,
                "memory_after_step": global_working_memory
            })

            step += 1

        print("\n[Warning] Max steps reached without Action 4. Returning the latest available prediction.")
        if latest_decision:
            return latest_decision, step_records
        else:
            return {"error": "Failed to generate any valid JSON prediction."}, step_records
