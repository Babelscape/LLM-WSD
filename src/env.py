import os
from dotenv import load_dotenv

load_dotenv()

gpt_key = os.getenv("OPENAI_API_KEY")
deepseek_key = os.getenv("DEEPSEEK_KEY")
hcp_path_to_models = os.getenv("HCP_PATH")