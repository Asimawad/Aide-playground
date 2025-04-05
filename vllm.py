# from openai import OpenAI
# client = OpenAI(
#     base_url="http://localhost:8000/v1",
#     api_key="token-abc123",
# )

# completion = client.chat.completions.create(
#   model="NousResearch/Meta-Llama-3-8B-Instruct",
#   messages=[
#     {"role": "user", "content": "Hello!"}
#   ]
# )

# print(completion.choices[0].message)
import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(0))
print(torch.cuda.memory_summary(device=None, abbreviated=False))