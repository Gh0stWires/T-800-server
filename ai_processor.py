from openai import OpenAI
from config import LLM_CONFIG
import chromadb
import autogen
import os
import time
import requests
import sys
import json
import datetime
from dotenv import load_dotenv

load_dotenv()

chroma_client = chromadb.PersistentClient(path=os.getenv("CHROMA_DB_PATH"))
chat_collection = chroma_client.get_or_create_collection(os.getenv("CHROMA_COLLECTION"))

client = OpenAI(base_url=os.getenv("LLM_API_BASE"), api_key="lm-studio")

def get_embedding(text, model="nomic-ai/nomic-embed-text-v1.5-GGUF"):
    """Generate an embedding for the given text using local model."""
    text = text.replace("\n", " ")  # Ensure clean input
    return client.embeddings.create(input=[text], model=model).data[0].embedding

# Create the T-800 AI Agent
terminator_agent = autogen.AssistantAgent(
    name="T800",
    llm_config=LLM_CONFIG,
    system_message=(
        "You are a Terminator AI. Speak in plain text. "
        "Do NOT return JSON, lists, or structured objects. "
        "Only respond in full sentences, staying in character."
    )
)

def web_search(query):
    """Perform a web search using Brave Search API and return structured results."""
    url = f"https://api.search.brave.com/res/v1/web/search?q={query}&count=5"
    headers = {"Accept": "application/json", "X-Subscription-Token": os.getenv("BRAVE_API_KEY")}

    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        return f"Error: Unable to fetch results ({response.status_code})"

    data = response.json()

    # ✅ Debugging: Print the raw JSON response
    #print("[DEBUG] Brave API Raw Response:")
    #print(json.dumps(data, indent=2))

    # ✅ Ensure the "web" field exists and contains valid search results
    if "web" not in data or "results" not in data["web"]:
        return "No relevant search results found."

    # ✅ Extract and format search results
    search_results = data["web"]["results"]

    parsed_results = []
    for result in search_results[:5]:  # Limit to top 5 results
        title = result.get("title", "No Title")
        url = result.get("url", "No URL")
        description = result.get("description", "No Description")

        parsed_results.append(f"Title: {title}\nURL: {url}\nSummary: {description}\n")

    return "\n".join(parsed_results) if parsed_results else "No relevant search results found."

def store_message(user_id, role, content):
    """Store a message in ChromaDB along with its embedding."""
    embedding = get_embedding(content)  # Generate embedding

    chat_collection.add(
        documents=[content],  # Store message content
        metadatas=[{"user_id": user_id, "role": role}],  # Store metadata
        embeddings=[embedding],  # Store embedding for vector search
        ids=[f"{user_id}_{datetime.datetime.now().timestamp()}"],  # Unique ID
    )


def retrieve_memory(user_id, question, num_matches=3):
    """Retrieve relevant past messages from ChromaDB using embeddings similarity search."""
    question_embedding = get_embedding(question)  # Get embedding for the new question

    # ✅ Query for similar past messages
    results = chat_collection.query(
        query_embeddings=[question_embedding],
        n_results=num_matches,
        where={"user_id": user_id}
    )

    # ✅ Ensure "documents" exists and is not None
    past_messages = results.get("documents", [])

    # ✅ Flatten list (if empty, return an empty string)
    past_messages = [msg[0] if isinstance(msg, list) and msg else "" for msg in past_messages]

    return "\n".join(past_messages).strip() if past_messages else ""  # Return empty string if no history

def should_perform_web_search(question):
    """Determine if a web search is required based on the question."""
    search_decision_prompt = f"""
    You are a highly intelligent AI with access to both memory and web searches.

    A user has asked the following question:
    "{question}"

    Your task:
    - Determine if a web search is required.
    - If the question is about recent events (e.g., news, sports results, latest data), answer "YES".
    - If the question is about general knowledge or historical facts, answer "NO".
    - Do NOT answer the question itself.
    - Your response must be exactly "YES" or "NO".

    **Output Format (strictly follow this)** 
    ```
    [DECISION] YES or NO
    ```
    """

    decision_response = terminator_agent.generate_reply(
        messages=[{"role": "user", "content": search_decision_prompt}],
        config_list=[{"max_tokens": 5, "temperature": 0}]  # ✅ Forces "YES" or "NO"
    )

    if isinstance(decision_response, str):
        decision_result = decision_response.strip().upper()
    elif isinstance(decision_response, autogen.ChatResult):
        decision_messages = [
            msg["content"].strip().upper() for msg in decision_response.chat_history if msg["role"] == "assistant"
        ]
        decision_result = decision_messages[-1] if decision_messages else "YES"  # Default to "YES" if AI doesn't answer correctly
    else:
        decision_result = "YES"

    print(f"[DEBUG] AI Search Decision: {decision_result}")
    
    # Ensure valid output (if AI gives a malformed response, default to "YES")
    return "YES" if decision_result not in ["YES", "NO"] else decision_result



def refine_search_query(question):
    """Generate an optimized search query for web search."""
    search_query_prompt = f"""
    You are an AI that specializes in refining search queries.
    Your ONLY task is to rewrite the given user query into a better, short, web search query.

    A user has asked the following:
    "{question}"

    - Do **NOT** answer the question.
    - Do **NOT** return explanations.
    - Do **NOT** add unrelated information.
    - Your response **must be EXACTLY ONE search query** (max 8 words).
    - Your query should be focused and concise.

    Example:
    User: "What's the latest news in the US?"
    Response: "latest US news today"

    **Output Format:**
    ```
    [SEARCH_QUERY] optimized query here
    ```
    """

    refined_search_query = terminator_agent.generate_reply(
        messages=[{"role": "user", "content": search_query_prompt}],
        config_list=[{"max_tokens": 10, "temperature": 0}]  # ✅ Forces short output
    )

    # ✅ Extract the refined search query
    if isinstance(refined_search_query, autogen.ChatResult):
        refined_queries = [
            msg["content"] for msg in refined_search_query.chat_history if msg["role"] == "assistant"
        ]
        refined_search_query = refined_queries[-1] if refined_queries else question  # Default to original question

    # ✅ Ensure the AI-generated search query is valid
    if "[SEARCH_QUERY]" in refined_search_query:
        refined_search_query = refined_search_query.replace("[SEARCH_QUERY]", "").strip()
    else:
        refined_search_query = question  # Fallback to original if format fails

    print(f"[DEBUG] AI-Generated Search Query: {refined_search_query}")
    return refined_search_query



def generate_response(user_id, question, refined_search_query, search_results, conversation_context):
    """Generate a response using AI memory and web search results, separating thinking and actual response."""
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    context = (
        f"Current Date and Time: {current_time}\n\n"
        "You are a T800 Terminator having a conversation with a human.\n"
        "Speak in clear, natural language and do NOT return code blocks, Python scripts, or shell commands.\n"
        "Stay in character as a Terminator, but keep responses text-based.\n\n"
        f"Relevant past context from previous interactions:\n{conversation_context}\n\n"
    )

    if search_results:
        context += f"Web search performed for: {refined_search_query}\n"
        context += f"Web Search Results:\n{search_results}\n\n"
    else:
        context += "No web search performed.\n\n"

    context += f"User's Question: {question}\n\n"
    context += "Use the available memory and search results (if any) to provide an answer."

    response = terminator_agent.generate_reply(
        messages=[{"role": "user", "content": context}],
        config_list=[{"max_tokens": 250, "temperature": 0.7}]
    )

    thinking = ""
    actual_response = ""

    if isinstance(response, dict):
        try:
            content = response["choices"][0]["message"]["content"].strip()
            if "<think>" in content and "</think>" in content:
                thinking = content.split("<think>")[1].split("</think>")[0].strip()
                actual_response = content.split("</think>")[1].strip()
            else:
                actual_response = content
        except (KeyError, IndexError):
            actual_response = "Error: Malformed response from LLM."

    elif isinstance(response, str):
        actual_response = response.strip()

    elif isinstance(response, autogen.ChatResult):
        if response.chat_history:
            for msg in reversed(response.chat_history):
                if msg["role"] == "assistant":
                    content = msg["content"].strip()
                    if "<think>" in content and "</think>" in content:
                        thinking = content.split("<think>")[1].split("</think>")[0].strip()
                        actual_response = content.split("</think>")[1].strip()
                    else:
                        actual_response = content

    if not actual_response:
        actual_response = "Error: No valid response from AI."

    return {"thinking": thinking, "response": actual_response}






def ask_t800(user_id, question):
    """Main function to handle AI responses and determine whether to perform a web search."""
    if not question.strip():
        return "Error: No input provided."

    # ✅ Step 1: Decide if a web search is needed
    search_needed = should_perform_web_search(question)
    print(f"[DEBUG] AI Search Decision: {search_needed}")

    search_results = ""
    refined_search_query = ""
    conversation_context = ""

    # ✅ Step 2: Perform a web search if needed
    if "YES" in search_needed:
        refined_search_query = refine_search_query(question)
        print(f"[DEBUG] AI-Generated Search Query: {refined_search_query}")

        search_results = web_search(refined_search_query)
        time.sleep(1)
        print(f"[DEBUG] Web Search Results: {search_results}")

        # 🛑 Important: DO NOT use old memory if we did a search!
        conversation_context = ""
    else:
        # ✅ Retrieve past conversation memory if NO web search was done
        conversation_context = retrieve_memory(user_id, question, num_matches=3)

    # ✅ Step 3: Generate AI response
    response = generate_response(user_id, question, refined_search_query, search_results, conversation_context)

    # ✅ Step 4: Store conversation in memory
    store_message(user_id, "user", question)
    store_message(user_id, "assistant", response)

    print("[DEBUG] Returning from ask_t800")
    return response.strip()