import streamlit as st
from azure.identity import ClientSecretCredential
from azure.keyvault.secrets import SecretClient
import requests
import time
import base64

# 从Streamlit Secrets获取保密信息
tenant_id = st.secrets["tenant_id"]
client_id = st.secrets["client_id"]
client_secret = st.secrets["client_secret"]
vault_url = st.secrets["vault_url"]

# 从Azure Key Vault获取Secret
def retrieve(secret_name):
    return SecretClient(vault_url=vault_url, credential=ClientSecretCredential(tenant_id, client_id, client_secret)).get_secret(secret_name).value

EXCELLENCE_API_KEY = retrieve("ExcellenceKey")
EXCELLENCE_ENDPOINT = retrieve("ExcellenceEndpoint")
EXCELLENCE2_API_KEY = retrieve("Excellence2Key")
EXCELLENCE2_ENDPOINT = retrieve("Excellence2Endpoint")
OPENROUTER_API_KEY = retrieve("OpenRouter")
DASHSCOPE_API_KEY = retrieve("DashScope")
MISTRAL_API_KEY = retrieve("Mistral")
AI21_API_KEY = retrieve("AI21")
ZHIPU_API_KEY = retrieve("Zhipu")
YI_API_KEY = retrieve("Lingyiwanwu")
MINIMAX_API_KEY = retrieve("MiniMax")
DEEPSEEK_API_KEY = retrieve("DeepSeek")
YUSI_PHI_VISION_API_KEY = retrieve("YusiPhiKey")
YUSI_PHI_VISION_ENDPOINT = retrieve("YusiPhiEndpoint")
COHERE_API_KEY = retrieve("Cohere")

# 发送请求并获取结果
def request_llm(url, headers, data, timeout=180, delay=1):
    for attempt in range(5):
        try:
            print(f"Sending request to {url}")
            response = requests.post(url, headers=headers, json=data, timeout=timeout).json()
            content = response.get('choices', [{}])[0].get('message', {}).get('content')
            if content:
                print(content)
                return content
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(delay)
            delay *= 2

    print("Failed to get a valid response after maximum retries")
    return None


# 请求Azure OpenAI
def yusi_excellence(system_message, user_message):
    url = f"{EXCELLENCE_ENDPOINT}/openai/deployments/yusi-excellence/chat/completions?api-version=2024-05-01-preview"
    headers = {
        "api-key": EXCELLENCE_API_KEY
    }
    data = {
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ],
        "temperature": 0.55,
        "top_p": 0.9,
        "max_tokens": 4000
    }
    return request_llm(url, headers, data)


# 请求Azure OpenAI
def yusi_mini(system_message, user_message):
    url = f"{EXCELLENCE2_ENDPOINT}/openai/deployments/yusi-mini/chat/completions?api-version=2024-05-01-preview"
    headers = {
        "api-key": EXCELLENCE2_API_KEY
    }
    data = {
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ],
        "temperature": 0.4,
        "top_p": 0.9,
        "max_tokens": 4000
    }
    return request_llm(url, headers, data)


# 请求OpenRouter
def openrouter(system_message, user_message, model, temperature, top_p, max_tokens):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}"
    }
    data = {
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ],
        "model": model,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens
    }
    return request_llm(url, headers, data)

def o1_preview_openrouter(system_message, user_message):
    return openrouter(system_message, user_message, "openai/o1-preview-2024-09-12", 0.55, 0.9, 4000)

def o1_mini_openrouter(system_message, user_message):
    return openrouter(system_message, user_message, "openai/o1-mini-2024-09-12", 0.55, 0.9, 4000)

def gpt_latest_openrouter(system_message, user_message):
    return openrouter(system_message, user_message, "openai/chatgpt-4o-latest", 0.55, 0.9, 4000)

def gpt_0806_openrouter(system_message, user_message):
    return openrouter(system_message, user_message, "openai/gpt-4o-2024-08-06", 0.55, 0.9, 4000)

def gpt_mini_openrouter(system_message, user_message):
    return openrouter(system_message, user_message, "openai/gpt-4o-mini-2024-07-18", 0.4, 0.9, 4000)

def claude_sonnet_openrouter(system_message, user_message):
    return openrouter(system_message, user_message, "anthropic/claude-3.5-sonnet", 0.55, 0.9, 4000)

def gemini_pro_openrouter(system_message, user_message):
    return openrouter(system_message, user_message, "google/gemini-pro-1.5-exp", 0.55, 0.9, 4000)

def gemini_flash_openrouter(system_message, user_message):
    return openrouter(system_message, user_message, "google/gemini-flash-1.5-exp", 0.4, 0.9, 4000)

def llama405b_openrouter(system_message, user_message):
    return openrouter(system_message, user_message, "meta-llama/llama-3.1-405b-instruct", 0.55, 0.9, 4000)

def llama70b_openrouter(system_message, user_message):
    return openrouter(system_message, user_message, "meta-llama/llama-3.1-70b-instruct", 0.5, 0.9, 4000)

def perplexity_openrouter(system_message, user_message):
    return openrouter(system_message, user_message, "perplexity/llama-3.1-sonar-large-128k-online", 0.4, 0.9, 4000)


# 请求DashScope
def dashscope(system_message, user_message, model, temperature, top_p, max_tokens):
    url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {DASHSCOPE_API_KEY}"
    }
    data = {
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ],
        "model": model,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens
    }
    return request_llm(url, headers, data)

def qwen72b_dashscope(system_message, user_message):
    return dashscope(system_message, user_message, "qwen2.5-72b-instruct", 0.5, 0.9, 4000)


# 请求Mistral
def mistral_large(system_message, user_message):
    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}"
    }
    data = {
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ],
        "model": "mistral-large-2407",
        "temperature": 0.5,
        "top_p": 0.9,
        "max_tokens": 4000
    }
    return request_llm(url, headers, data)


# 请求AI21
def jamba(system_message, user_message, model, temperature, top_p, max_tokens):
    url = "https://api.ai21.com/studio/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {AI21_API_KEY}"
    }
    data = {
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ],
        "model": model,
        "temperature": temperature,
        "top_p": top_p,
        "maxTokens": max_tokens
    }
    return request_llm(url, headers, data)

def jamba_large(system_message, user_message):
    return jamba(system_message, user_message, "jamba-1.5-large", 0.55, 0.9, 4000)

def jamba_mini(system_message, user_message):
    return jamba(system_message, user_message, "jamba-1.5-mini", 0.4, 0.9, 4000)


# 请求Zhipu
def glm(system_message, user_message, model, temperature, top_p, max_tokens):
    url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    headers = {
        "Authorization": f"Bearer {ZHIPU_API_KEY}"
    }
    data = {
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ],
        "model": model,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens
    }
    return request_llm(url, headers, data)

def glm_plus(system_message, user_message):
    return glm(system_message, user_message, "glm-4-plus", 0.5, 0.9, 4000)


# 请求Lingyiwanwu
def yi(system_message, user_message, model, temperature, top_p, max_tokens):
    url = "https://api.lingyiwanwu.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {YI_API_KEY}"
    }
    data = {
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ],
        "model": model,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens
    }
    return request_llm(url, headers, data)

def yi_large(system_message, user_message):
    return yi(system_message, user_message, "yi-large", 0.5, 0.9, 4000)


# 请求MiniMax
def abab(system_message, user_message):
    url = "https://api.minimax.chat/v1/text/chatcompletion_v2"
    headers = {
        "Authorization": f"Bearer {MINIMAX_API_KEY}"
    }
    data = {
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ],
        "model": "abab6.5s-chat",
        "temperature": 0.5,
        "top_p": 0.9,
        "max_tokens": 4000
    }
    return request_llm(url, headers, data)


# 请求DeepSeek
def deepseek(system_message, user_message):
    url = "https://api.deepseek.com/chat/completions"
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
    }
    data = {
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ],
        "model": "deepseek-chat",
        "temperature": 0.5,
        "top_p": 0.9,
        "max_tokens": 4000
    }
    return request_llm(url, headers, data)


# 请求OpenRouter
def vision_openrouter(message, image_path, model):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}"
    }
    data = {
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": message},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{base64.b64encode(open(image_path, 'rb').read()).decode()}"}}]}
        ],
        "model": model
    }
    return request_llm(url, headers, data)

def pixtral_openrouter(message, image_path):
    return vision_openrouter(message, image_path, "mistralai/pixtral-12b:free")

def qwen_vision_openrouter(message, image_path):
    return vision_openrouter(message, image_path, "qwen/qwen-2-vl-72b-instruct")


# 请求Zhipu
def glm_vision(message, image_path):
    url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    headers = {
        "Authorization": f"Bearer {ZHIPU_API_KEY}"
    }
    data = {
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": message},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{base64.b64encode(open(image_path, 'rb').read()).decode()}"}}]}
        ],
        "model": "glm-4v-plus"
    }
    return request_llm(url, headers, data)


# 请求Lingyiwanwu
def yi_vision(message, image_path):
    url = "https://api.lingyiwanwu.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {YI_API_KEY}"
    }
    data = {
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": message},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{base64.b64encode(open(image_path, 'rb').read()).decode()}"}}]}
        ],
        "model": "yi-vision"
    }
    return request_llm(url, headers, data)


# 请求Azure AI
def phi_vision(message, image_path):
    url = f"{YUSI_PHI_VISION_ENDPOINT}/chat/completions"
    headers = {
        "Authorization": f"Bearer {YUSI_PHI_VISION_API_KEY}"
    }
    data = {
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": message},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{base64.b64encode(open(image_path, 'rb').read()).decode()}"}}]}
        ],
    }
    return request_llm(url, headers, data)


# 向Cohere发送请求并获取结果
def request_cohere(url, headers, data, timeout=180, delay=1):
    for attempt in range(5):
        try:
            print(f"Sending request to {url}")
            response = requests.post(url, headers=headers, json=data, timeout=timeout).json()
            content = response.get('text')
            if content:
                print(content)
                return content
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(delay)
            delay *= 2

    print("Failed to get a valid response after maximum retries")
    return None

# 请求Cohere
def command(system_message, user_message, model, temperature, p, max_tokens):
    url = "https://api.cohere.com/v1/chat"
    headers = {
        "Authorization": f"Bearer {COHERE_API_KEY}"
    }
    data = {
        "preamble": system_message,
        "message": user_message,
        "prompt_truncation": "AUTO",
        "connectors": [{"id": "web-search"}],
        "model": model,
        "temperature": temperature,
        "p": p,
        "max_tokens": max_tokens
    }
    return request_cohere(url, headers, data)

def command_r_plus(system_message, user_message):
    return command(system_message, user_message, "command-r-plus-08-2024", 0.55, 0.9, 4000)

def command_r(system_message, user_message):
    return command(system_message, user_message, "command-r-08-2024", 0.4, 0.9, 4000)
