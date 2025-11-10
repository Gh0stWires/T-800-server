from openai import OpenAI
import chromadb
import os
import json
from ai_util import get_developer_message
from ai_util import get_system_message
from datetime import datetime

from dotenv import load_dotenv
from openai_harmony import (
    load_harmony_encoding,
    HarmonyEncodingName,
    Conversation,
    Message,
    Role,
    DeveloperContent,
    Author,
    StreamableParser
)

load_dotenv()

# ------ CONFIG ------
# last known working model qwen3-8b-64k-josiefied-uncensored-neo-max
DEFAULT_AGENT_NAME = "Miss Minutes"
MAX_RECENT_TURNS = 8          # How many turns to include after the summary
SUMMARIZE_AFTER = 30          # How many messages before we summarize
MODEL_ID = "openai/gpt-oss-20b"  # Model to use for chat completions


#harmony_encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

client = OpenAI(base_url=os.getenv("LLM_API_BASE"), api_key="lm-studio")

chroma_client = chromadb.PersistentClient(path=os.getenv("CHROMA_DB_PATH"))
chat_collection = chroma_client.get_or_create_collection(os.getenv("CHROMA_COLLECTION"))


def get_embedding(text, model="nomic-ai/nomic-embed-text-v1.5-GGUF"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding


def store_message(user_id, role, content):
    embedding = get_embedding(content)
    chat_collection.add(
        documents=[content],
        metadatas=[{"user_id": user_id, "role": role}],
        embeddings=[embedding],
        ids=[f"{user_id}_{datetime.datetime.now().timestamp()}"]
    )


def build_system_prompt(agent_name, user_id, system_prompt=None):
    name_intro = f"You are {agent_name}."
    user_info = f" The user's ID is {user_id}."
    
    if system_prompt:
        return f"{name_intro} {system_prompt.strip()} {user_info}"
    
    return (
        f"{name_intro} An advanced, witty, and helpful female Southern AI assistant. "
        "Your goal is to serve and delight the user in a real conversation, just like two people chatting. "
        "Do not write scene directions, stage directions, or narrate your actions. "
        "Do not role-play. Do not write as if you are in a play or script. "
        "Speak directly to the user in first-person, naturally, and with Southern charm."
        f"{user_info}"
    )




def retrieve_memory_with_summary(user_id, num_recent=MAX_RECENT_TURNS):
    results = chat_collection.get(
        where={"user_id": user_id},
        include=["documents", "metadatas"]
    )
    history = []
    for doc, meta, _id in zip(results['documents'], results['metadatas'], results['ids']):
        try:
            timestamp = float(_id.split("_")[-1])
        except Exception:
            timestamp = 0
        role = meta.get("role", "user")
        msg = doc[0] if isinstance(doc, list) and doc else doc
        history.append((timestamp, role, msg, _id))
    history = sorted(history, key=lambda x: x[0])

    summary = None
    recent = []
    for (timestamp, role, msg, _id) in history:
        if role == "summary":
            summary = msg
            recent = []  # Only messages *after* the latest summary count as "recent"
        else:
            recent.append((role, msg))
    # Only keep the last N recent messages
    return summary, recent[-num_recent:]


def count_user_messages(user_id):
    results = chat_collection.get(
        where={"user_id": user_id},
        include=["metadatas"]
    )
    return sum(1 for meta in results['metadatas'] if meta.get("role") != "summary")



def summarize_chat_history(user_id, agent_name=DEFAULT_AGENT_NAME, num_to_summarize=SUMMARIZE_AFTER):
    # Get all messages for this user
    results = chat_collection.get(
        where={"user_id": user_id},
        include=["documents", "metadatas"]
    )
    history = []
    for doc, meta, _id in zip(results['documents'], results['metadatas'], results['ids']):
        try:
            timestamp = float(_id.split("_")[-1])
        except Exception:
            timestamp = 0
        role = meta.get("role", "user")
        msg = doc[0] if isinstance(doc, list) and doc else doc
        history.append((timestamp, role, msg, _id))
    history = sorted(history, key=lambda x: x[0])
    to_summarize = history[:num_to_summarize]
    if not to_summarize:
        return None

    summary_prompt = (
        f"You are {agent_name}, an advanced assistant. Summarize the following conversation "
        "so that you remember the important facts, topics, preferences, and any emotional tone. "
        "Summarize for yourself, as notes to help future responses. Be concise.\n\n"
    )
    for _, role, msg, _ in to_summarize:
        summary_prompt += f"{role.title()}: {msg}\n"

    summary_response = client.chat.completions.create(
        model=MODEL_ID,
        messages=[
            {"role": "system", "content": summary_prompt}
        ],
        temperature=0.2,
        max_tokens=300
    )
    summary = summary_response.choices[0].message.content.strip()

    store_message(user_id, "summary", summary)

    # Delete the old, summarized messages
    ids_to_delete = [_id for (_, _, _, _id) in to_summarize]
    if ids_to_delete:
        chat_collection.delete(ids=ids_to_delete)
    return summary


def ask_ai(user_id, question, agent_name=DEFAULT_AGENT_NAME, system_prompt_override=None):
    #store_message(user_id, "user", question) 
    total = count_user_messages(user_id)
    print(total)
    """ if total >= SUMMARIZE_AFTER:
        summarize_chat_history(user_id, agent_name) """

    #summary, conversation_history = retrieve_memory_with_summary(user_id)
    system_prompt = build_system_prompt(agent_name, user_id, system_prompt_override)

    messages = [{"role": "system", "content": system_prompt}]
    """ if summary:
        messages.append({"role": "system", "content": f"Summary of earlier conversation: {summary}"})
    for role, content in conversation_history:
        messages.append({"role": role, "content": content}) """
    messages.append({"role": "user", "content": question})

    response_iter = client.chat.completions.create(
        model=MODEL_ID,
        messages=messages,
        temperature=0.7,
        max_tokens=2500,
        stream=True
    )

    buffer = ""
    found_think = False
    full_response = ""

    for chunk in response_iter:
        delta = chunk.choices[0].delta
        if hasattr(delta, "content") and delta.content:
            buffer += delta.content
            full_response += delta.content
            if not found_think and "<think>" in buffer and "</think>" in buffer:
                before_think, after_think = buffer.split("<think>", 1)
                think_content, after = after_think.split("</think>", 1)
                yield {"type": "thinking", "content": think_content.strip()}
                if after.strip():
                    yield {"type": "response", "content": after}
                found_think = True
                buffer = ""
            elif found_think:
                yield {"type": "response", "content": delta.content}
    #store_message(user_id, "assistant", full_response.strip())



def ask_open_gpt(user_id, question, agent_name=DEFAULT_AGENT_NAME, system_prompt_override=None, fromVoice=False):
    system_instruction = system_prompt_override or (
        "You are a helpful assistant. Respond with concise and polite answers."
    )

    voice_response_schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "voice_response",
            "schema": {
                "type": "object",
                "properties": {
                    "voice_output": {
                        "type": "string",
                        "description": "A short and concise sentence suitable for speaking to the user."
                    }
                },
                "required": ["voice_output"]
            }
        }
    }

    params = {
    "model": MODEL_ID,
    "messages": build_open_gpt_messages(
        question,
        system_identity=system_instruction,
        user_id=user_id,
        agent_name=agent_name,
        fromVoice=fromVoice
    ),
    "stream": True,
    "temperature": 0.9,
    "max_tokens": 2500,
    "stream_options": {"include_usage": True}
    }


    
    # Send tokens directly using OpenAI-compatible `messages` API
    response = client.chat.completions.create(**params)

    
    
    for chunk in response:
        if not getattr(chunk, "choices", None):
            continue
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if getattr(delta, "reasoning", None):
            yield {"type": "thinking", "content": delta.reasoning}
        if getattr(delta, "content", None):
            yield {"type": "response", "content": delta.content}


def build_open_gpt_messages(user_message, system_identity=None, user_id=None, agent_name=DEFAULT_AGENT_NAME, date=None, fromVoice=False):
    """
    Build a list of messages for OpenGPT with system and user, reasoning set to HIGH.
    Returns a list of dicts suitable for OpenAI/chat API.
    """
    if not system_identity:
        system_identity = "You are ChatGPT, a large language model trained by OpenAI."
    if not date:
        date = datetime.now().strftime("%Y-%m-%d")
    system_msg = get_system_message(system_identity, date, user_id=user_id, agent_name=agent_name, fromVoice=fromVoice)
    messages = [
        {"role": "system", "content": str(system_msg)},
        {"role": "user", "content": user_message}
    ]
    return messages

# Example usage for CLI/debug
if __name__ == "__main__":
    user_id = "default_user"
    agent_name = "Miss Minutes"
    user_input = input(f"Ask {agent_name}: ")
    store_message(user_id, "user", user_input)
    for chunk in ask_ai(user_id, user_input, agent_name):
        print(chunk)
