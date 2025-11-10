from openai_harmony import (
    Author,
    Conversation,
    DeveloperContent,
    HarmonyEncodingName,
    Message,
    Role,
    SystemContent,
    ToolDescription,
    load_harmony_encoding,
    ReasoningEffort
)

def get_system_message(identity, date, reasoning_effort="high", knowledge_cutoff="2024-06", user_id=None, agent_name=None, fromVoice=False):
    """
    Returns a dict for the system role message for OpenAI/chat API, with a fully constructed prompt.
    """
    
    prompt = f"You are {agent_name}. {identity} Reasoning effort: {reasoning_effort}. date: {date}. Knowledge cutoff: {knowledge_cutoff}. Conversation With: {user_id}"
    if fromVoice:
        prompt += " When responding, keep responses very short and with no formatting"
    return {
        "role": "system",
        "content": prompt
    }

def get_developer_message(system_instruction):
    return (
        DeveloperContent.new()
            .with_instructions(system_instruction)
    )

"""
            .with_function_tools(
                [
                    ToolDescription.new(
                        "get_location",
                        "Gets the location of the user.",
                    ),
                    ToolDescription.new(
                        "get_current_weather",
                        "Gets the current weather in the provided location.",
                        parameters={
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "The city and state, e.g. San Francisco, CA",
                                },
                                "format": {
                                    "type": "string",
                                    "enum": ["celsius", "fahrenheit"],
                                    "default": "celsius",
                                },
                            },
                            "required": ["location"],
                        },
                    ),
                ]
            ) """
