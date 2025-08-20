import os
from zai import ZhipuAiClient
from dotenv import load_dotenv
load_dotenv()

# 从环境变量读取 API Key
api_key = os.getenv("ZHIPUAI_API_KEY")
if not api_key:
    raise ValueError("请先设置环境变量 ZHIPUAI_API_KEY")
# 可选：从环境变量读取 BASE_URL
base_url = os.getenv("ZHIPUAI_BASE_URL", None)

# Initialize client
client = ZhipuAiClient(api_key=api_key, base_url=base_url ) 

# Create chat completion
response = client.chat.completions.create(
    model="glm-4.5",
    messages=[
        {"role": "user", "content": "你好"}
    ]
)
#print(response.choices[0].message.content)
print(response)