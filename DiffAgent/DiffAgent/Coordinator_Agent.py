import json
import re
import time
from openai import OpenAI

client = OpenAI(
    api_key="your key",  # Replace with your key
    base_url="url"  # Replace with your llm url
)
MODEL_NAME = "model_name"  # Replace with your llm name, such as DeepSeek V3


def build_text_messages(system_prompt, content):
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content}
    ]


def llm_analysis(system_prompt, content, model=MODEL_NAME):
    messages = build_text_messages(system_prompt, content)
    start_time = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        extra_body={
            "chat_template_kwargs": {"enable_thinking": True}
        }
    )
    time_span = time.perf_counter() - start_time
    return time_span, response.choices[0].message.content

class CoordinatorAgent:
    def __init__(self, semantic_expert, profile_expert, topology_expert, max_steps=6):
        print("Initializing Coordinator Agent...")
        self.semantic_expert = semantic_expert
        self.profile_expert = profile_expert
        self.topology_expert = topology_expert
        self.max_steps = max_steps

        self.system_prompt = """You are the Coordinator Agent within the DiffAgent architecture.
Your primary responsibility is to orchestrate an asynchronous cognitive reasoning loop to predict information diffusion paths in social networks.

Initial State Constraints:
You are only provided with the target [Topic_ID] and a list of [Candidate_User_IDs]. You do not have direct access to the raw topic content, user profiles, or social topology.

Action Space:
To bridge the information gap and construct the Global Working Memory, you must dynamically invoke the following expert agents:

Action 1: Call Semantic Expert
Function: Assigns the Semantic Expert to parse the underlying multimodal diffusion motivations of the topic.
Action_Input: {"topic_id": "<ID>"}

Action 2: Call Profile Expert
Function: Assigns the Profile Expert to evaluate individual susceptibility by aligning user history with the parsed topic semantics.
Action_Input: {"user_ids": [<ID_1>, <ID_2>...]}

Action 3: Call Topology Expert
Function: Assigns the Topology Expert to quantify structural social attraction within the current cascade.
Action_Input: {"user_ids": [<ID_1>, <ID_2>...]}

Action 4: Final Decision
Function: Terminates the reasoning loop when the working memory contains sufficient, conflict-free evidence. Outputs the final consensus ranking and logical explanation.
Action_Output: {"ranked_user_ids": [...], "summary": "<Your transparent logical explanation>"}

Output Formatting:
For every iteration, you must strictly output your cognitive state in the following format:
Thought: [Analyze the current Memory. Identify missing evidence Gap, or perform reflection to resolve conflicting evidence and recalibrate weights.]
Action: [Select ONE action from the Action Space: 'Action 1', 'Action 2', 'Action 3', or 'Action 4']
Action_Output: [Provide the corresponding parameters in valid JSON format]"""

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
        global_working_memory = "Memory is currently empty.\n"
        step = 1

        parsed_topic_semantics = ""

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
Based on the current state, generate your next Thought, Action, and Action_Output."""

            _, llm_response = llm_analysis(self.system_prompt, user_prompt)

            parsed = self.parse_llm_response(llm_response)
            action_type = parsed["Action"]
            action_params = self.extract_json(parsed["Action_Output"])

            print(f"Thought: {parsed['Thought'][:100]}...")
            print(f"Action Executed: {action_type}")

            if "Action 1" in action_type:
                target_topic = action_params.get("topic_id", topic_id)
                print(f"-> System implicitly calling Semantic Expert for Topic {target_topic}...")

                result = self.semantic_expert.analyze_topic(target_topic)
                parsed_topic_semantics = result.get("content", "Error fetching semantic data.")

                global_working_memory += f"[Observation Step {step} - Semantic Data]: {parsed_topic_semantics}\n"

            elif "Action 2" in action_type:
                target_users = action_params.get("user_ids", candidate_uids)
                print(f"-> System implicitly calling Profile Expert for Users {target_users}...")

                if not parsed_topic_semantics:
                    observation = "[System Error]: Cannot run Profile Expert without Parsed Topic Semantics. Call Action 1 first."
                else:
                    result = self.profile_expert.evaluate_susceptibility(parsed_topic_semantics, target_users)
                    observation = result.get("content", "Error fetching profile data.")

                global_working_memory += f"[Observation Step {step} - Susceptibility Profiles]: {observation}\n"

            elif "Action 3" in action_type:
                target_users = action_params.get("user_ids", candidate_uids)
                print(f"-> System implicitly calling Topology Expert for Users {target_users}...")

                result = self.topology_expert.evaluate_topology(cascade_uids, target_users)
                observation = result.get("content", "Error fetching topology data.")

                global_working_memory += f"[Observation Step {step} - Topological Attraction]: {observation}\n"

            elif "Action 4" in action_type:
                print("-> Final Decision Reached!")
                final_decision = action_params
                return final_decision

            else:
                global_working_memory += f"[System Error]: Invalid action '{action_type}'. Please select Action 1, 2, 3, or 4.\n"

            step += 1

        print("\n[Warning] Max steps reached without Action 4.")
        return {"error": "Failed to reach a consensus within max steps."}