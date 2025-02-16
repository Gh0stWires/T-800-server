import datetime
import autogen
from flask import Flask, request, jsonify
from config import LLM_CONFIG

app = Flask(__name__)

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

# User Proxy Agent
user_proxy = autogen.UserProxyAgent(
    name="UserProxy",
    human_input_mode="ALWAYS",
    code_execution_config=False,
    description="Handles interaction with the user.",
)

def ask_t800(question):
    """Handles chatbot responses."""
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    context = f"""
    Current Date and Time: {current_time}

    You are a T800 Terminator having a conversation with a human.
    Speak in clear, natural language and do NOT return code blocks, Python scripts, or shell commands.
    Stay in character as a Terminator, but keep responses text-based.

    Question: {question}
    """

    response = user_proxy.initiate_chat(
        terminator_agent,
        message=context,
        max_turns=1
    )

    # Extract text if response is a ChatResult
    if isinstance(response, autogen.ChatResult):
        response = response.content

    return response.strip() if isinstance(response, str) else "Error: Response format invalid."

@app.route("/chat", methods=["POST"])
def chat():
    """Receives chat requests from the Android app."""
    data = request.get_json()
    question = data.get("message", "")

    if not question.strip():
        return jsonify({"error": "Empty message"}), 400

    response = ask_t800(question)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
