import httpx
import asyncio
import json

STREAM_URL = "http://10.0.0.86:1234/v1/chat/completions"

async def stream_chat():
    payload = {
        "model": "openai/gpt-oss-20b",
        "messages": [
            {"role": "system", "content": "Always answer in rhymes."},
            {"role": "user", "content": "Introduce yourself."}
        ],
        "temperature": 0.7,
        "max_tokens": 250,
        "stream": False
    }

    assistant_response = []

    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", STREAM_URL, json=payload) as response:
            print(f"Status code: {response.status_code}")
            if response.status_code != 200:
                print("Error:", await response.aread())
                return

            async for line_bytes in response.aiter_lines():
                line = line_bytes.strip()
                if not line:
                    continue
                if line == "data: [DONE]":
                    print("\nStream finished.")
                    break

                if line.startswith("data:"):
                    data_str = line[len("data:"):].strip()
                    try:
                        data = json.loads(data_str)
                        delta = data["choices"][0]["delta"]
                        if "role" in delta:
                            # Usually only in the first chunk
                            print(f"\nRole: {delta['role']}")
                        if "content" in delta:
                            content = delta["content"]
                            assistant_response.append(content)
                            print(content, end="", flush=True)
                    except Exception as e:
                        print(f"Failed to parse JSON chunk: {e}")

    full_response = "".join(assistant_response)
    print("\n\nFull assistant response:")
    print(full_response)


if __name__ == "__main__":
    asyncio.run(stream_chat())
