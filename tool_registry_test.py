import json
from agent_tools import AVAILABLE_TOOLS, TOOL_DEFINITIONS

# 1. SETUP: This is the file the "Agent" wants to check
# (Update this to a real image path on your cluster)
TEST_IMAGE_PATH = "./FairVision/Glaucoma/Training/slo_fundus_00001.jpg"

print("--- 🤖 SIMULATING GPT-4 AGENT ---")

# 2. AGENT THINKING PROCESS
print(f"Agent Thought: 'The user suspects Glaucoma. I see a tool called tool_glaucoma_check in my registry.'")
print(f"Agent Action: Generating function call for '{TEST_IMAGE_PATH}'...")

# 3. AGENT GENERATES CALL (Simulated JSON from OpenAI)
gpt_response_mock = {
    "name": "tool_glaucoma_check",
    "arguments": json.dumps({"image_path": TEST_IMAGE_PATH})
}

# 4. SYSTEM EXECUTES TOOL
tool_name = gpt_response_mock["name"]
tool_args = json.loads(gpt_response_mock["arguments"])

if tool_name in AVAILABLE_TOOLS:
    print(f"\n>> System: executing {tool_name} with args {tool_args}...")
    
    # --- THE MAGIC MOMENT ---
    function_to_call = AVAILABLE_TOOLS[tool_name]
    observation = function_to_call(**tool_args)
    # ------------------------
    
    print("\n--- 📝 OBSERVATION RECEIVED BY AGENT ---")
    parsed_obs = json.loads(observation)
    print(json.dumps(parsed_obs, indent=2))
    
    if parsed_obs['status'] == 'success':
        print(f"\n>> Agent Final Response: 'The structural analysis confirms a high risk. The Cup-to-Disc Ratio is {parsed_obs['vertical_cdr']}, which is significant.'")
else:
    print("Tool not found!")