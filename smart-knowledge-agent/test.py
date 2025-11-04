import openai

openai.api_key = "sk-or-v1-..."

resp = openai.chat.completions.create(
    model="openai/gpt-4o-mini",
    messages=[{"role": "user", "content": "hi"}],
)
print(resp)
