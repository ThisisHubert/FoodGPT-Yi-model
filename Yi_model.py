import os
# pip install autogen
import autogen
from autogen import AssistantAgent, UserProxyAgent
from autogen.agentchat.contrib.math_user_proxy_agent import MathUserProxyAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
from autogen.code_utils import extract_code

config_list = autogen.config_list_from_json(
    "OAI_CONFIG_LIST",
    file_location=".",
)
if not config_list:
    os.environ["MODEL"] = "01-ai/Yi-34B-200K"
    os.environ["OPENAI_API_KEY"] = "<your openai api key>"
    os.environ["OPENAI_BASE_URL"] = "<your openai base url>" # optional

    config_list = autogen.config_list_from_models(
        model_list=[os.environ.get("MODEL", "gpt-35-turbo")],
    )

llm_config = {
    "timeout": 60,
    "cache_seed": 42,
    "config_list": config_list,
    "temperature": 0,
}

def termination_msg(x):
    _msg = str(x.get("content", "")).upper().strip().strip("\n").strip(".")
    return isinstance(x, dict) and (_msg.endswith("TERMINATE") or _msg.startswith("TERMINATE"))

def _is_termination_msg(message):
    if isinstance(message, dict):
        message = message.get("content")
        if message is None:
            return False
    cb = extract_code(message)
    contain_code = False
    for c in cb:
        # todo: support more languages
        if c[0] == "python":
            contain_code = True
            break
    return not contain_code

agents = []

# Filling parameters
agent = UserProxyAgent(
    name="User_Proxy",
    is_termination_msg=termination_msg,
    human_input_mode="TERMINATE",
    system_message="""""",
    default_auto_reply="Thank you. Reply `TERMINATE` to finish.",
    max_consecutive_auto_reply=5,
    code_execution_config=False,
)


agents.append(agent)


agent = AssistantAgent(
    name="Assistant_Agent",
    system_message="""You are a helpful AI assistant, able to understand pictures and texts, suggest food that boost the users health and how to preserve food products properly with the aim to reduce food waste in households globally.
In the following cases, give the users details about the food ingredients.
    1. When you need to collect info based on the user's question, browse or search the web, get the expired date if possible, . After sufficient info is obtained and the task is ready to be solved based.
    2. When users are asking question based on the uploaded picture, give explanation in detailed about the food and suggest how to preserve the food properly to prevent food waste. Finish the task smartly.
Solve the task step by step if you need to. If a plan is not provided, explain your plan first. Be clear with it!
If you want the user to save the suggestions in a file before asking another question, you can ask the user to reply "SAVE" in the end. The user will be asked to provide a file name. The file will be saved in the same directory as this script.
Reply "TERMINATE" in the end when everything is done.
    """,
    llm_config=llm_config,
    is_termination_msg=termination_msg,
)


agents.append(agent)


init_sender = None
for agent in agents:
    if "UserProxy" in str(type(agent)):
        init_sender = agent
        break

if not init_sender:
    init_sender = agents[0]


recipient = agents[1] if agents[1] != init_sender else agents[0]

if isinstance(init_sender, (RetrieveUserProxyAgent, MathUserProxyAgent)):
    init_sender.initiate_chat(recipient, problem="I would love to have an ice cream today, if I want to buy a few cups and store it in the fridge, how long is acceptable?")
else:
    init_sender.initiate_chat(recipient, message="I would love to have an ice cream today, if I want to buy a few cups and store it in the fridge, how long is acceptable?")
