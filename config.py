import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Configure the LLM to use your local API
CONFIG_LIST = [
    {
        "model": "gpt-4-turbo",
        "base_url": os.getenv("LLM_API_BASE", "http://localhost:6666/v1"),
        "api_key": os.getenv("LLM_API_KEY", "not-needed"),
    }
]

LLM_CONFIG = {
    "config_list": CONFIG_LIST,
    "temperature": 0.7,
    "stream": False,
}
