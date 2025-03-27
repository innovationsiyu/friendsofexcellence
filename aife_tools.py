from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
import azure.cognitiveservices.speech as speechsdk
import os
import requests
import importlib
from itertools import permutations
import base64
import mimetypes
import pandas as pd
import json
from pathvalidate import sanitize_filename
import Levenshtein
import random
import fitz
import ast
import wave
import math
from decimal import Decimal, getcontext
from scraper import tidy_body_contents, get_web_contents, get_web_texts
from request_web_search import tavily_by_freshness, bing_by_freshness, bing_by_year_range, google_by_freshness, google_by_year_range, exa_by_freshness, exa_by_year_range
from search_results_to_word import search_results_to_word, append_company_info_and_disclaimer
from aife_utils import retrieve, REMOVE_FILETYPE, manage_futures, manage_futures_async, upload_to_container, upload_and_email_file, resize_images, pdf_to_images, extract_text_with_xml_tag, extract_text_with_tags, extract_text_with_pattern, filter_words, now_and_choices


OPENROUTER_API_KEY = retrieve("OpenRouter")
HYPERBOLIC_API_KEY = retrieve("Hyperbolic")
DEEPINFRA_API_KEY = retrieve("DeepInfra")
DASHSCOPE_API_KEY = retrieve("DashScope")
SILICONFLOW_API_KEY = retrieve("SiliconFlow")
LUCHEN_API_KEY = retrieve("Luchen")
LUCHEN2_API_KEY = retrieve("Luchen2")
DEEPSEEK_API_KEY = retrieve("DeepSeek")
YI_API_KEY = retrieve("Lingyiwanwu")
MINIMAX_API_KEY = retrieve("MiniMax")
XAI_API_KEY = retrieve("xAI")
EXCELLENCE_API_KEY = retrieve("ExcellenceKey")
EXCELLENCE_ENDPOINT = retrieve("ExcellenceEndpoint")
EXCELLENCE2_API_KEY = retrieve("Excellence2Key")
EXCELLENCE2_ENDPOINT = retrieve("Excellence2Endpoint")
R1_API_KEY = retrieve("DeepSeekR1Key")

DOCUMENT_CONFIGS = [{"api_key": retrieve("YusiMultiKey"), "endpoint": retrieve("YusiMultiEndpoint")},
                   {"api_key": retrieve("YusiMulti2Key"), "endpoint": retrieve("YusiMulti2Endpoint")}]

SPEECH_CONFIGS = [{"api_key": retrieve("SpeechKey"), "region": "eastus"},
                  {"api_key": retrieve("Speech2Key"), "region": "westus2"}]


def execute(tool_calls):
    try:
        requests = []
        for tool_call in tool_calls:
            if function := tool_call.get("function"):
                if (name := function.get("name")) and (arguments := function.get("arguments")):
                    requests.append((globals()[name], json.loads(arguments)))
        return {f"{function.__name__}({arguments})": result for result, function, arguments in manage_futures(requests)}
    except Exception as e:
        print(f"Failed to execute tool calls: {e}")
        return None


def request_llm(url, headers, data):
    for attempt in range(2):
        try:
            print(f"Sending request to {url} with {data}")
            response = requests.post(url, headers=headers, json=data, timeout=120).json()
            print(response)
            if message := response.get("choices", [{}])[0].get("message", {}):
                if tool_calls := message.get("tool_calls"):
                    if tool_results := execute(tool_calls):
                        return {"type": "tool_results", "tool_results": f"<tool_results>{tool_results}</tool_results>"}
                elif content := message.get("content"):
                    if reasoning := message.get("reasoning_content") or message.get("reasoning"):
                        content = f"<think>{reasoning}</think>\n\n{content}"
                    if citations := response.get("citations"):
                        content = f"{content}\n\n{'\n\n'.join(f'[{i + 1}] {citation}' for i, citation in enumerate(citations))}"
                    return content
            raise Exception(f"Invalid response structure: {response}")
        except Exception as e:
            print(f"Request LLM attempt {attempt + 1} failed: {e}")
    return None


class LLM:
    def __init__(self, url, api_key):
        self.url = url
        self.api_key = api_key

    def __call__(self, messages, model=None, temperature=None, top_p=None, max_tokens=None, tools=None):
        headers = {
            "Authorization": f"Bearer {self.api_key}"
        }
        data = {
            "messages": messages,
            **({"model": model} if model else {}),
            **({"temperature": temperature} if temperature is not None else {}),
            **({"top_p": top_p} if top_p else {}),
            **({"max_tokens": max_tokens} if max_tokens else {}),
            **({"tools": tools} if tools else {})
        }
        return request_llm(self.url, headers, data)


class AzureOpenAI:
    def __init__(self, endpoint, api_key):
        self.endpoint = endpoint
        self.api_key = api_key

    def __call__(self, messages, model, temperature=None, top_p=None, max_tokens=None, tools=None):
        url = f"{self.endpoint}openai/deployments/{model}/chat/completions?api-version=2024-10-21"
        headers = {
            "api-key": self.api_key
        }
        data = {
            "messages": messages,
            **({"temperature": temperature} if temperature is not None else {}),
            **({"top_p": top_p} if top_p else {}),
            **({"max_tokens": max_tokens} if max_tokens else {}),
            **({"tools": tools} if tools else {})
        }
        return request_llm(url, headers, data)


openrouter = LLM("https://openrouter.ai/api/v1/chat/completions", OPENROUTER_API_KEY)
hyperbolic = LLM("https://api.hyperbolic.xyz/v1/chat/completions", HYPERBOLIC_API_KEY)
deepinfra = LLM("https://api.deepinfra.com/v1/openai/chat/completions", DEEPINFRA_API_KEY)
dashscope = LLM("https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions", DASHSCOPE_API_KEY)
siliconflow = LLM("https://api.siliconflow.cn/v1/chat/completions", SILICONFLOW_API_KEY)
luchen = LLM("https://cloud.luchentech.com/api/maas/chat/completions", LUCHEN_API_KEY)
luchen2 = LLM("https://cloud.luchentech.com/api/maas/chat/completions", LUCHEN2_API_KEY)
deepseek = LLM("https://api.deepseek.com/chat/completions", DEEPSEEK_API_KEY)
lingyiwanwu = LLM("https://api.lingyiwanwu.com/v1/chat/completions", YI_API_KEY)
minimax = LLM("https://api.minimax.chat/v1/text/chatcompletion_v2", MINIMAX_API_KEY)
xai = LLM("https://api.x.ai/v1/chat/completions", XAI_API_KEY)
excellence = AzureOpenAI(EXCELLENCE_ENDPOINT, EXCELLENCE_API_KEY)
excellence2 = AzureOpenAI(EXCELLENCE2_ENDPOINT, EXCELLENCE2_API_KEY)
r1 = LLM("https://DeepSeek-R1-yusi.westus3.models.ai.azure.com/chat/completions", R1_API_KEY)


def get_prompt(name, **arguments):
    prompt = getattr(importlib.import_module(f"aife_prompts.{name}"), name)
    if arguments:
        if callable(prompt):
            return prompt(**arguments)
        else:
            return prompt.format(**arguments)
    else:
        if callable(prompt):
            return prompt()
        else:
            return prompt


def get_tools(tools):
    if tools:
        return [getattr(importlib.import_module("aife_tools"), tool) for tool in tools]
    return None


def route_llm_request(configs, messages, temperature=None, top_p=None, tools=None):
    for config in configs:
        try:
            llm_config = llm_configs[config]
            name = llm_config["name"]
            arguments = {**llm_config["arguments"]}
            if temperature is not None:
                arguments["temperature"] = temperature
            if top_p:
                arguments["top_p"] = top_p
            if llm_result := globals()[name](messages, **arguments, tools=tools):
                return llm_result
        except Exception:
            continue
    return None


def text_chat(configs, system_message, user_message, index=0):
    if user_message:
        messages = [{"role": "system", "content": system_message}, {"role": "user", "content": user_message}]
        return route_llm_request(configs, messages)
    else:
        return None


def text_chats(configs, system_message, user_messages):
    configs = {i: list(permutation) for i, permutation in enumerate(permutations(configs))}
    requests = [(text_chat, (configs[i % len(configs)], system_message, user_message, i)) for i, user_message in enumerate(user_messages if isinstance(user_messages, list) else [user_messages], start=0)]
    return {arguments[-1]: result for result, function, arguments in manage_futures(requests)}


def image_chat(configs, user_message, image_path):
    if image_path:
        messages = [{"role": "user", "content": [{"type": "text", "text": user_message}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64.b64encode(open(image_path, 'rb').read()).decode('utf-8')}"}}]}]
        return route_llm_request(configs, messages)
    else:
        return None


def image_chats(configs, user_message, image_paths):
    configs = {i: list(permutation) for i, permutation in enumerate(permutations(configs))}
    requests = [(image_chat, (configs[i % len(configs)], user_message, image_path)) for i, image_path in enumerate(image_paths if isinstance(image_paths, list) else [image_paths], start=0)]
    return {arguments[-1]: result for result, function, arguments in manage_futures(requests)}


def request_stt(url, headers, files, data):
    for attempt in range(2):
        try:
            print(f"Sending request to {url}")
            response = requests.post(url, headers=headers, files=files, data=data, timeout=30).json()
            if text := response.get("text"):
                return text
            raise Exception(f"Invalid response: {response}")
        except Exception as e:
            print(f"Request STT attempt {attempt + 1} failed: {e}")
    return None


def speech_to_text(file_path):
    url = "https://api.deepinfra.com/v1/openai/audio/transcriptions"
    headers = {
        "Authorization": f"Bearer {DEEPINFRA_API_KEY}"
    }
    files = {
        "file": open(file_path, "rb")
    }
    data = {
        "model": "openai/whisper-large-v3-turbo"
    }
    return request_stt(url, headers, files, data)


jobs = {
    "Text Chat": {
        "category": "text",
        "llms": ["DeepSeek-V3-0324", "Qwen-Max", "DeepSeek-R1", "QwQ 32B", "MiniMax-01", "Gemini Pro 2.5 Experimental", "Gemini Flash 2.0", "Gemma 3 27B", "Mistral Small 3.1", "Phi 4", "Claude 3.7 Sonnet", "o3 mini high", "GPT-4o", "Grok 2", "Yi-Lightning", "LFM 7B"],
        "system_message": "chat_only",
        "tools": None,
        "intro": "体验全球第一梯队的大语言模型"
    },
    "Text Chat with Search": {
        "category": "text",
        "llms": ["GPT-4o-mini Search", "DeepSeek-R1 Search", "DeepSeek-R1 Research"],
        "system_message": "perplexity_chat_and_search",
        "tools": None,
        "intro": "支持搜索和深度研究"
    },
    "Multimodal Chat with Tools": {
        "category": "multimodal",
        "llms": ["MiniMax-01", "Gemini Pro 2.5 Experimental", "Gemini Flash 2.0", "Claude 3.7 Sonnet", "GPT-4o"],
        "system_message": "chat_with_tools",
        "tools": ["get_web_texts_func", "generate_audio_interpretation_func", "calculator_func"],
        "intro": "读网页 生成文档解读音频 使用计算器"
    },
    "Translater": {
        "category": "text",
        "llms": ["DeepSeek-V3-0324", "Qwen-Max", "DeepSeek-R1", "QwQ 32B", "MiniMax-01", "Gemini Pro 2.5 Experimental", "Gemini Flash 2.0", "Gemma 3 27B", "Mistral Small 3.1", "Phi 4", "Claude 3.7 Sonnet", "o3 mini high", "GPT-4o", "Grok 2", "Yi-Lightning", "LFM 7B"],
        "system_message": "language_translation_and_optimisation",
        "tools": None,
        "intro": "翻译和语言优化任务专用"
    },
    "Logic Checker": {
        "category": "text",
        "llms": ["DeepSeek-V3-0324", "Qwen-Max", "DeepSeek-R1", "QwQ 32B", "MiniMax-01", "Gemini Pro 2.5 Experimental", "Gemini Flash 2.0", "Gemma 3 27B", "Mistral Small 3.1", "Phi 4", "Claude 3.7 Sonnet", "o3 mini high", "GPT-4o", "Grok 2", "Yi-Lightning", "LFM 7B"],
        "system_message": "logic_checking_and_critical_thinking",
        "tools": None,
        "intro": "逻辑检查和批判性思考专用"
    }
}

llms = {
    "QwQ 32B": {
        "configs": ["qwq_32b_deepinfra", "qwq_32b_hyperbolic", "qwq_32b_openrouter"],
        "context_length": 131072
    },
    "Qwen-Max": {
        "configs": ["qwen_max_dashscope"],
        "context_length": 32768
    },
    "DeepSeek-R1": {
        "configs": ["deepseek_r1_azure", "deepseek_r1_hyperbolic", "deepseek_r1", "deepseek_r1_dashscope", "deepseek_r1_luchen2", "deepseek_r1_luchen"],
        "context_length": 65536
    },
    "DeepSeek-V3-0324": {
        "configs": ["deepseek_v3_openrouter", "deepseek_v3_hyperbolic", "deepseek_v3_deepinfra", "deepseek_v3"],
        "context_length": 65536
    },
    "GPT-4o-mini Search": {
        "configs": ["gpt4o_mini_search_openrouter"],
        "context_length": 128000
    },
    "DeepSeek-R1 Search": {
        "configs": ["sonar_reasoning_pro_openrouter"],
        "context_length": 128000
    },
    "DeepSeek-R1 Research": {
        "configs": ["sonar_deep_research_openrouter"],
        "context_length": 128000
    },
    "MiniMax-01": {
        "configs": ["minimax_01", "minimax_01_openrouter"],
        "context_length": 1000192
    },
    "Gemini Pro 2.5 Experimental": {
        "configs": ["gemini_25_pro_openrouter"],
        "context_length": 1000192
    },
    "Gemini Flash 2.0": {
        "configs": ["gemini2_flash_openrouter"],
        "context_length": 1000192
    },
    "Gemma 3 27B": {
        "configs": ["gemma3_27b_openrouter"],
        "context_length": 131072
    },
    "Mistral Small 3.1": {
        "configs": ["mistral_small_31_openrouter"],
        "context_length": 128000
    },
    "Phi 4": {
        "configs": ["phi4_deepinfra", "phi4_openrouter"],
        "context_length": 16384
    },
    "Grok 2": {
        "configs": ["grok2_xai", "grok2_openrouter"],
        "context_length": 131072
    },
    "o3 mini high": {
        "configs": ["o3_mini_high_openrouter"],
        "context_length": 200000
    },
    "GPT-4o": {
        "configs": ["gpt4o_excellence", "gpt4o_openrouter"],
        "context_length": 128000
    },
    "Claude 3.7 Sonnet": {
        "configs": ["claude37_sonnet_openrouter"],
        "context_length": 200000
    },
    "Yi-Lightning": {
        "configs": ["yi_lightning"],
        "context_length": 16384
    },
    "LFM 7B": {
        "configs": ["lfm_7b_openrouter"],
        "context_length": 32768
    }
}

llm_configs = {
    "qwq_32b_hyperbolic": {
        "name": "hyperbolic",
        "arguments": {
            "model": "Qwen/QwQ-32B",
            "temperature": 0.15,
            "top_p": 0.95,
            "max_tokens": 8192
        }
    },
    "qwq_32b_deepinfra": {
        "name": "deepinfra",
        "arguments": {
            "model": "Qwen/QwQ-32B",
            "temperature": 0.15,
            "top_p": 0.95,
            "max_tokens": 8192
        }
    },
    "qwq_32b_openrouter": {
        "name": "openrouter",
        "arguments": {
            "model": "qwen/qwq-32b",
            "temperature": 0.15,
            "top_p": 0.95,
            "max_tokens": 8192
        }
    },
    "qwen_max_dashscope": {
        "name": "dashscope",
        "arguments": {
            "model": "qwen-max-2025-01-25",
            "temperature": 0.15,
            "top_p": 0.95,
            "max_tokens": 8192
        }
    },
    "deepseek_r1_hyperbolic": {
        "name": "hyperbolic",
        "arguments": {
            "model": "deepseek-ai/DeepSeek-R1",
            "temperature": 0.15,
            "top_p": 0.95,
            "max_tokens": 8192
        }
    },
    "deepseek_r1": {
        "name": "deepseek",
        "arguments": {
            "model": "deepseek-reasoner",
            "temperature": 0.15,
            "top_p": 0.95,
            "max_tokens": 8192
        }
    },
    "deepseek_r1_azure": {
        "name": "r1",
        "arguments": {
            "temperature": 0.15,
            "top_p": 0.95,
            "max_tokens": 8192
        }
    },
    "deepseek_r1_dashscope": {
        "name": "dashscope",
        "arguments": {
            "model": "deepseek-r1",
            "temperature": 0.15,
            "top_p": 0.95,
            "max_tokens": 8192
        }
    },
    "deepseek_r1_luchen": {
        "name": "luchen",
        "arguments": {
            "model": "VIP/deepseek-ai/DeepSeek-R1",
            "temperature": 0.15,
            "top_p": 0.95,
            "max_tokens": 8192
        }
    },
    "deepseek_r1_luchen2": {
        "name": "luchen2",
        "arguments": {
            "model": "VIP/deepseek-ai/DeepSeek-R1",
            "temperature": 0.15,
            "top_p": 0.95,
            "max_tokens": 8192
        }
    },
    "deepseek_v3_hyperbolic": {
        "name": "hyperbolic",
        "arguments": {
            "model": "deepseek-ai/DeepSeek-V3-0324",
            "temperature": 0.15,
            "top_p": 0.95,
            "max_tokens": 8192
        }
    },
    "deepseek_v3_deepinfra": {
        "name": "deepinfra",
        "arguments": {
            "model": "deepseek-ai/DeepSeek-V3-0324",
            "temperature": 0.15,
            "top_p": 0.95,
            "max_tokens": 8192
        }
    },
    "deepseek_v3_openrouter": {
        "name": "openrouter",
        "arguments": {
            "model": "deepseek/deepseek-chat-v3-0324",
            "temperature": 0.15,
            "top_p": 0.95,
            "max_tokens": 8192
        }
    },
    "deepseek_v3": {
        "name": "deepseek",
        "arguments": {
            "model": "deepseek-chat",
            "temperature": 0.15,
            "top_p": 0.95,
            "max_tokens": 8192
        }
    },
    "sonar_deep_research_openrouter": {
        "name": "openrouter",
        "arguments": {
            "model": "perplexity/sonar-deep-research",
            "temperature": 0.15,
            "top_p": 0.95,
            "max_tokens": None
        }
    },
    "sonar_reasoning_pro_openrouter": {
        "name": "openrouter",
        "arguments": {
            "model": "perplexity/sonar-reasoning-pro",
            "temperature": 0.15,
            "top_p": 0.95,
            "max_tokens": 8192
        }
    },
    "gpt4o_mini_search_openrouter": {
        "name": "openrouter",
        "arguments": {
            "model": "openai/gpt-4o-mini-search-preview",
            "temperature": 0.15,
            "top_p": 0.95,
            "max_tokens": 8192
        }
    },
    "gemma3_27b_openrouter": {
        "name": "openrouter",
        "arguments": {
            "model": "google/gemma-3-27b-it",
            "temperature": 0.15,
            "top_p": 0.95,
            "max_tokens": 8192
        }
    },
    "mistral_small_31_openrouter": {
        "name": "openrouter",
        "arguments": {
            "model": "mistralai/mistral-small-3.1-24b-instruct-2503",
            "temperature": 0.15,
            "top_p": 0.95,
            "max_tokens": 8192
        }
    },
    "grok2_xai": {
        "name": "xai",
        "arguments": {
            "model": "grok-2-1212",
            "temperature": 0.15,
            "top_p": 0.95,
            "max_tokens": 8192
        }
    },
    "grok2_openrouter": {
        "name": "openrouter",
        "arguments": {
            "model": "x-ai/grok-2-1212",
            "temperature": 0.15,
            "top_p": 0.95,
            "max_tokens": 8192
        }
    },
    "o3_mini_high_openrouter": {
        "name": "openrouter",
        "arguments": {
            "model": "openai/o3-mini-high",
            "temperature": 0.15,
            "top_p": 0.95,
            "max_tokens": 8192
        }
    },
    "gpt4o_openrouter": {
        "name": "openrouter",
        "arguments": {
            "model": "openai/gpt-4o-2024-08-06",
            "temperature": 0.15,
            "top_p": 0.95,
            "max_tokens": 8192
        }
    },
    "gpt4o_excellence": {
        "name": "excellence",
        "arguments": {
            "model": "excellence",
            "temperature": 0.15,
            "top_p": 0.95,
            "max_tokens": 8192
        }
    },
    "claude37_sonnet_openrouter": {
        "name": "openrouter",
        "arguments": {
            "model": "anthropic/claude-3.7-sonnet:thinking",
            "temperature": 0.15,
            "top_p": 0.95,
            "max_tokens": 8192
        }
    },
    "gemini_25_pro_openrouter": {
        "name": "openrouter",
        "arguments": {
            "model": "google/gemini-2.5-pro-exp-03-25:free",
            "temperature": 0.15,
            "top_p": 0.95,
            "max_tokens": 8192
        }
    },
    "gemini2_flash_openrouter": {
        "name": "openrouter",
        "arguments": {
            "model": "google/gemini-2.0-flash-001",
            "temperature": 0.15,
            "top_p": 0.95,
            "max_tokens": 8192
        }
    },
    "minimax_01": {
        "name": "minimax",
        "arguments": {
            "model": "MiniMax-Text-01",
            "temperature": 0.15,
            "top_p": 0.95,
            "max_tokens": 8192
        }
    },
    "minimax_01_openrouter": {
        "name": "openrouter",
        "arguments": {
            "model": "minimax/minimax-01",
            "temperature": 0.15,
            "top_p": 0.95,
            "max_tokens": 8192
        }
    },
    "phi4_deepinfra": {
        "name": "deepinfra",
        "arguments": {
            "model": "microsoft/phi-4",
            "temperature": 0.15,
            "top_p": 0.95,
            "max_tokens": 8192
        }
    },
    "phi4_openrouter": {
        "name": "openrouter",
        "arguments": {
            "model": "microsoft/phi-4",
            "temperature": 0.15,
            "top_p": 0.95,
            "max_tokens": 8192
        }
    },
    "yi_lightning": {
        "name": "lingyiwanwu",
        "arguments": {
            "model": "yi-lightning",
            "temperature": 0.15,
            "top_p": 0.95,
            "max_tokens": 8192
        }
    },
    "lfm_7b_openrouter": {
        "name": "openrouter",
        "arguments": {
            "model": "liquid/lfm-7b",
            "temperature": 0.15,
            "top_p": 0.95,
            "max_tokens": 8192
        }
    },
    "gpt4o_mini_openrouter": {
        "name": "openrouter",
        "arguments": {
            "model": "openai/gpt-4o-mini-2024-07-18",
            "temperature": 0.15,
            "top_p": 0.95,
            "max_tokens": 8192
        }
    },
    "gpt4o_mini_excellence": {
        "name": "excellence2",
        "arguments": {
            "model": "yusi-mini",
            "temperature": 0.15,
            "top_p": 0.95,
            "max_tokens": 8192
        }
    },
    "qwen_vl_plus_dashscope": {
        "name": "dashscope",
        "arguments": {
            "model": "qwen-vl-plus-2025-01-25",
            "temperature": 0.15,
            "top_p": 0.95,
            "max_tokens": None
        }
    },
    "qwen25_vl_7b_dashscope": {
        "name": "dashscope",
        "arguments": {
            "model": "qwen2.5-vl-7b-instruct",
            "temperature": 0.15,
            "top_p": 0.95,
            "max_tokens": None
        }
    },
    "qwen25_vl_7b_hyperbolic": {
        "name": "hyperbolic",
        "arguments": {
            "model": "Qwen/Qwen2.5-VL-7B-Instruct",
            "temperature": 0.15,
            "top_p": 0.95,
            "max_tokens": None
        }
    },
    "qwen_vl_max_dashscope": {
        "name": "dashscope",
        "arguments": {
            "model": "qwen-vl-max-2025-01-25",
            "temperature": 0.15,
            "top_p": 0.95,
            "max_tokens": None
        }
    },
    "qwen25_vl_72b_dashscope": {
        "name": "dashscope",
        "arguments": {
            "model": "qwen2.5-vl-72b-instruct",
            "temperature": 0.15,
            "top_p": 0.95,
            "max_tokens": None
        }
    },
    "qwen25_vl_72b_hyperbolic": {
        "name": "hyperbolic",
        "arguments": {
            "model": "Qwen/Qwen2.5-VL-72B-Instruct",
            "temperature": 0.15,
            "top_p": 0.95,
            "max_tokens": None
        }
    }
}

information_retrieval_func = {
    "type": "function",
    "function": {
        "name": "information_retrieval",
        "description": "Utilise search engines to retrieve information by applying professional search strategies. Call this when systematic structuring of information in specified areas of expertise and time ranges is required.",
        "parameters": {
            "type": "object",
            "properties": {
                "search_strategies": {
                    "type": "array",
                    "description": "Multiple search strategies as a list of dictionaries, with each one including a distinct search target, acceptance criteria, required quantity, query, and freshness or year range.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "search_target": {
                                "type": "string",
                                "description": "The specification of the objects or answers to search for."
                            },
                            "acceptance_criteria": {
                                "type": "string",
                                "description": "The criteria that an acceptable search result should meet. This will be used for filtering the search results."
                            },
                            "required_quantity": {
                                "type": "integer",
                                "description": "The minimum number of acceptable search results required. A further search will be conducted if the filtered search results are insufficient."
                            },
                            "query": {
                                "type": "string",
                                "description": "The input to search engines, which can be keywords or phrases, and may also include advanced search operators."
                            },
                            "freshness_or_year_range": {
                                "type": ["integer", "array", "null"],
                                "description": "The time range filter, which can be a 1-digit integer representing the number of days to look back, a pair of 4-digit integers representing the start year and end year, or 'null'.",
                                "oneOf": [
                                    {
                                        "type": "integer",
                                        "description": "The number of days to look back for filtering the results by published date. This is a 1-digit integer. The common values are 7 or 30. Omit this if no specific recency is required.",
                                    },
                                    {
                                        "type": "array",
                                        "description": "The start year and end year for filtering the results by published date. This is a pair of 4-digit integers, in which the first represents the start year and the second represents the end year. Example: [2022, 2025]. Omit this if no specific year range is required.",
                                        "items": {
                                            "type": "integer",
                                            "description": "A 4-digit integer representing a year."
                                        },
                                        "minItems": 2,
                                        "maxItems": 2
                                    },
                                    {
                                        "type": "null"
                                    }
                                ]
                            }
                        },
                        "required": ["search_target", "acceptance_criteria", "required_quantity", "query"],
                        "additionalProperties": False
                    },
                    "minItems": 1
                }
            },
            "required": ["search_strategies"],
            "additionalProperties": False
        }
    }
}

export_search_results_func = {
    "type": "function",
    "function": {
        "name": "export_search_results",
        "description": "Process a CSV file containing search results, export it to a Word document, and return a URL for downloading it.",
        "parameters": {
            "type": "object",
            "properties": {
                "csv_path": {
                    "type": "string",
                    "description": "The path of the CSV file containing the search results.",
                },
            },
            "required": ["csv_path"],
            "additionalProperties": False,
        },
    }
}

get_web_texts_func = {
    "type": "function",
    "function": {
        "name": "get_web_texts",
        "description": "Scrape the full text of each webpage from the provided URLs. Call this when there are single or multiple URLs and the user requests or you need to browse one or more of the webpages.",
        "parameters": {
            "type": "object",
            "properties": {
                "web_urls": {
                    "type": "array",
                    "description": "A list of URLs to simultaneously scrape text from.",
                    "items": {
                        "type": "string",
                        "description": "The URL to scrape text from."
                    }
                }
            },
            "required": ["web_urls"],
            "additionalProperties": False
        }
    }
}

generate_audio_interpretation_func = {
    "type": "function",
    "function": {
        "name": "generate_audio_interpretation",
        "description": "Generate an audio interpretation for reports, plans, and studies. This function returns a URL for downloading the audio file.",
        "parameters": {
            "type": "object",
            "properties": {
                "txt_path": {
                    "type": "string",
                    "description": "The path of the TXT file containing the entire content of the documents. You can find this after the doc_content in the chat history."
                },
                "user_requirements": {
                    "type": "string",
                    "description": "The user's requirements for the interpretation."
                },
                "voice_gender": {
                    "type": "string",
                    "enum": ["male", "female"],
                    "description": "The user's choice of male or female voice for the audio."
                },
            },
            "required": ["txt_path", "user_requirements", "voice_gender"],
            "additionalProperties": False,
        },
    },
}

calculator_func = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Evaluate a wide range of mathematical expressions with high precision (50 decimal places). This supports basic arithmetic operations and math module functions.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The mathematical expression to evaluate. Examples: '2*3.14159', 'Decimal(1)/Decimal(3)', 'math.sqrt(2)', 'math.sin(math.pi/2)'",
                }
            },
            "required": ["expression"],
            "additionalProperties": False,
        },
    }
}


def remove_filetype_in_query(query):
    return REMOVE_FILETYPE.sub("", query)


def is_webpage(url):
    mime_type, encoding = mimetypes.guess_type(url)
    if mime_type and mime_type not in ("text/html", "application/xhtml+xml"):
        return False
    return True


def search_results_to_csv(csv_path, search_target, search_results):
    columns = ["search_target", "url", "title", "summary", "date"]
    df = pd.DataFrame([{**{"search_target": search_target}, **{column: search_result.get(column) for column in columns if column in search_result}} for search_result in search_results]).reindex(columns=columns)
    df = df.drop_duplicates(subset="url", keep="first")
    df = df[df["url"].apply(is_webpage)]
    df.to_csv(csv_path, index=False, encoding="utf-8")


def relevant_search_results_to_csv(csv_path, search_target):
    system_message = get_prompt("are_relevant_search_results", search_target=search_target)
    df = pd.read_csv(csv_path, encoding="utf-8")
    chunks = [df.iloc[i:i + 30] for i in range(0, len(df), 30)]
    if user_messages := [json.dumps(chunk[["title", "summary"]].to_dict("index"), indent=4, ensure_ascii=False) for chunk in chunks]:
        for attempt in range(3):
            try:
                llm_results = text_chats(["deepseek_v3_openrouter", "deepseek_v3_hyperbolic"], system_message, user_messages)
                if search_results := [search_result for i, indices in llm_results.items() for search_result in chunks[i].loc[chunks[i].index.isin(json.loads(indices))].to_dict("index").values()]:
                    columns = ["search_target", "url", "title", "date"]
                    df = pd.DataFrame([{column: search_result.get(column) for column in columns if column in search_result} for search_result in search_results]).reindex(columns=columns)
                    df.to_csv(csv_path, index=False, encoding="utf-8")
                    return True
            except Exception:
                continue
    return None


def accepted_search_results_to_csv(csv_path, search_target, acceptance_criteria):
    system_message = get_prompt("is_acceptable_search_result", search_target=search_target, acceptance_criteria=acceptance_criteria)
    df = pd.read_csv(csv_path, encoding="utf-8")
    if urls := (df[df["url"].notna() & df["url"].apply(bool)]["url"].tolist()):
        for attempt in range(3):
            try:
                web_texts = get_web_texts(urls)
                user_messages = [f"<web_text>{web_texts[url]}</web_text>" if web_texts[url] else None for url in urls]
                llm_results = text_chats(["gemini2_flash_openrouter"], system_message, user_messages)
                if search_results := [{**df.iloc[i].to_dict(), "web_text": web_texts[urls[i]]} for i in range(len(urls)) if (llm_result := llm_results[i]) and "False" not in llm_result and "false" not in llm_result]:
                    columns = ["search_target", "url", "title", "web_text", "date"]
                    df = pd.DataFrame([{column: search_result.get(column) for column in columns if column in search_result} for search_result in search_results]).reindex(columns=columns)
                    df.to_csv(csv_path, index=False, encoding="utf-8")
                    return True
            except Exception:
                continue
    return None


def gather_search_results(csv_path, existing_search_results=None):
    df = pd.read_csv(csv_path, encoding="utf-8")
    search_results = [{column: row[column] for column in df.columns if pd.notna(row.get(column))} for i, row in df.iterrows()]
    if existing_search_results:
        search_results.extend(existing_search_results)
        df = pd.DataFrame(search_results)
        df.to_csv(csv_path, index=False, encoding="utf-8")
    return search_results


def search_by_year_range(search_target, acceptance_criteria, query, year_range=None, existing_search_results=None):
    query = remove_filetype_in_query(query)
    requests = [
        (bing_by_year_range, (query, year_range)),
        (google_by_year_range, (query, year_range)),
        (tavily_by_freshness, (query,)),
        (exa_by_year_range, (query, year_range))
    ]
    if search_results := [item for result, function, arguments in manage_futures(requests) if result for item in result]:
        csv_path = os.path.join("temp-data", f"{sanitize_filename(search_target[:20])} {now_and_choices()}.csv")
        search_results_to_csv(csv_path, search_target, search_results)
        if relevant_search_results_to_csv(csv_path, search_target):
            if accepted_search_results_to_csv(csv_path, search_target, acceptance_criteria):
                return gather_search_results(csv_path, existing_search_results)
    return None


def search_by_freshness(search_target, acceptance_criteria, query, freshness=None, existing_search_results=None):
    query = remove_filetype_in_query(query)
    requests = [
        (bing_by_freshness, (query, freshness)),
        (google_by_freshness, (query, freshness)),
        (tavily_by_freshness, (query, freshness)),
        (exa_by_freshness, (query, freshness))
    ]
    if search_results := [item for result, function, arguments in manage_futures(requests) if result for item in result]:
        csv_path = os.path.join("temp-data", f"{sanitize_filename(search_target[:20])} {now_and_choices()}.csv")
        search_results_to_csv(csv_path, search_target, search_results)
        if relevant_search_results_to_csv(csv_path, search_target):
            if accepted_search_results_to_csv(csv_path, search_target, acceptance_criteria):
                return gather_search_results(csv_path, existing_search_results)
    return None


def endeavour_search(search_target, acceptance_criteria, required_quantity, query, freshness_or_year_range, system_message, existing_search_results=None, retry=0):
    if isinstance(freshness_or_year_range, list) and len(freshness_or_year_range) == 2:
        search_results = search_by_year_range(search_target, acceptance_criteria, query, freshness_or_year_range, existing_search_results)
    elif isinstance(freshness_or_year_range, int) or freshness_or_year_range is None:
        search_results = search_by_freshness(search_target, acceptance_criteria, query, freshness_or_year_range, existing_search_results)
    else:
        search_results = None
    if search_results and len(search_results) >= required_quantity or retry == 2:
        print(search_results)
        return search_results
    else:
        user_message = f"The filtered search results are:\n{json.dumps(search_results, indent=4, ensure_ascii=False)}\n\nThe query used to obtain these results is:\n{query}\n\nPlease revise the search target and query for the next search."
        for attempt in range(3):
            try:
                llm_result = text_chat(["gpt4o_mini_excellence"], system_message, user_message)
                if new_query := extract_text_with_xml_tag(llm_result, "new_query"):
                    return endeavour_search(search_target, acceptance_criteria, required_quantity, new_query, freshness_or_year_range, system_message, search_results, retry + 1)
            except Exception:
                continue
        return search_results


def all_gathered_search_results_to_csv(search_results):
    df = pd.DataFrame(search_results)
    df = df.drop_duplicates(subset="url", keep="first")
    csv_path = os.path.join("temp-data", f"{now_and_choices()}.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8")
    return csv_path


def information_retrieval(search_strategies):
    requests = []
    for search_strategy in search_strategies:
        if all((search_target := search_strategy.get("search_target"), acceptance_criteria := search_strategy.get("acceptance_criteria"), required_quantity := search_strategy.get("required_quantity"), query := search_strategy.get("query"))):
            freshness_or_year_range = search_strategy.get("freshness_or_year_range")
            system_message = get_prompt("revise_search_query", search_target=search_target, acceptance_criteria=acceptance_criteria)
            requests.append((endeavour_search, (search_target, acceptance_criteria, required_quantity, query, freshness_or_year_range, system_message)))
    if search_results := [item for result, function, arguments in manage_futures(requests) if result for item in result]:
        csv_path = all_gathered_search_results_to_csv(search_results)
        return json.dumps({"url_to_download_all_results": upload_to_container(csv_path)}, indent=4, ensure_ascii=False)
    return None


def levenshtein_similarity(s1, s2):
    return 1 - Levenshtein.distance(s1, s2) / max(len(s1), len(s2)) if max(len(s1), len(s2)) > 0 else 1


def match_first_line(extracted_paragraph, web_content):
    best_similarity = -1
    best_start_index = -1
    for i in range(len(web_content)):
        for combination in range(1, 4):
            if i + combination > len(web_content):
                continue
            similarity = levenshtein_similarity(extracted_paragraph, "".join(web_content[i:i + combination]))
            if similarity == 1:
                return i
            if similarity > best_similarity:
                best_similarity = similarity
                best_start_index = i
    return best_start_index


def match_last_line(extracted_paragraph, web_content):
    best_similarity = -1
    best_end_index = -1
    for i in range(len(web_content) - 1, -1, -1):
        for combination in range(1, 4):
            if i < combination - 1:
                continue
            similarity = levenshtein_similarity(extracted_paragraph, "".join(web_content[i - combination + 1:i + 1]))
            if similarity == 1:
                return i
            if similarity > best_similarity:
                best_similarity = similarity
                best_end_index = i
    return best_end_index


def extract_body_content(web_content, first_paragraph, last_paragraph):
    first_line_index = match_first_line(first_paragraph, web_content)
    last_line_index = match_last_line(last_paragraph, web_content)
    start_index = min(first_line_index, last_line_index)
    end_index = max(first_line_index, last_line_index)
    while start_index > 0 and any(web_content[i].startswith("temp-images") for i in range(max(0, start_index - 2), start_index)):
        start_index = next(i for i in range(start_index - 1, -1, -1) if web_content[i].startswith("temp-images"))
    while end_index + 1 < len(web_content) and any(web_content[i].startswith("temp-images") for i in range(end_index + 1, min(len(web_content), end_index + 3))):
        end_index = next(i for i in range(end_index + 1, len(web_content)) if web_content[i].startswith("temp-images"))
    if end_index + 1 < len(web_content) and web_content[end_index].startswith("temp-images"):
        end_index += 1
    return web_content[start_index:end_index + 1]


def get_structured_search_result(llm_result, web_content):
    if structured_search_result := extract_text_with_tags(extract_text_with_xml_tag(llm_result, "structured_info"), ["Title", "Source", "Published date", "First paragraph", "Last paragraph"]):
        return {"doc_title": structured_search_result["Title"], "source": structured_search_result["Source"], "published_date": structured_search_result["Published date"], "body_content": extract_body_content(web_content, structured_search_result["First paragraph"], structured_search_result["Last paragraph"])}
    return None


def structured_search_results_to_csv(csv_path):
    system_message = get_prompt("extract_structured_search_result")
    df = pd.read_csv(csv_path, encoding="utf-8")
    if urls := (df[df["url"].notna() & df["url"].apply(bool)]["url"].tolist()):
        for attempt in range(3):
            try:
                web_contents = get_web_contents(urls)
                user_messages = [f"<web_contents>{'\n'.join(web_contents[url])}</web_contents>" if web_contents[url] else None for url in urls]
                llm_results = text_chats(["deepseek_v3_openrouter", "deepseek_v3_hyperbolic"], system_message, user_messages)
                search_results = [{**df.iloc[i].to_dict(), **structured_search_result} if (llm_result := llm_results[i]) and len(llm_result) > 100 and (structured_search_result := get_structured_search_result(llm_result, web_contents[urls[i]])) else {**df.iloc[i].to_dict()} for i in range(len(urls))]
                columns = ["search_target", "url", "title", "doc_title", "source", "published_date", "body_content", "date"]
                df = pd.DataFrame([{column: search_result.get(column) for column in columns if column in search_result} for search_result in search_results]).reindex(columns=columns)
                df.to_csv(csv_path, index=False, encoding="utf-8")
                if sum(1 for result in search_results if "body_content" in result) == len(urls):
                    return True
                else:
                    return upload_to_container(csv_path)
            except Exception:
                continue
    return None


def ensure_structured_search_results(csv_path):
    df = pd.read_csv(csv_path, encoding="utf-8")
    df = df[df["body_content"].notna() & df["body_content"].apply(bool)]
    if manual_indices := [i for i in df.index if not (df.at[i, "body_content"].strip().startswith('[') and df.at[i, "body_content"].strip().endswith(']'))]:
        system_message = get_prompt("extract_structured_search_result")
        processed_body_contents = tidy_body_contents(df.loc[manual_indices, "body_content"].tolist())
        body_contents = [processed_body_contents[df.at[i, "body_content"]] for i in manual_indices]
        user_messages = [f"<web_contents>{body_contents[i]}</web_contents>" for i in range(len(manual_indices))]
        llm_results = text_chats(["deepseek_v3_openrouter", "deepseek_v3_hyperbolic"], system_message, user_messages)
        for i in range(len(manual_indices)):
            structured_search_result = get_structured_search_result(llm_results[i], body_contents[i])
            df.at[manual_indices[i], "doc_title"] = structured_search_result["doc_title"]
            df.at[manual_indices[i], "source"] = structured_search_result["source"]
            df.at[manual_indices[i], "published_date"] = structured_search_result["published_date"]
            df.at[manual_indices[i], "body_content"] = structured_search_result["body_content"]
    df.to_csv(csv_path, index=False, encoding="utf-8")
    return True


def export_search_results(csv_path):
    try:
        df = pd.read_csv(csv_path, encoding="utf-8")
        if not all(column in df.columns for column in ["doc_title", "source", "published_date", "body_content"]):
            result = structured_search_results_to_csv(csv_path)
            if isinstance(result, str):
                return result
            elif not result:
                return None
        if ensure_structured_search_results(csv_path):
            doc_path = search_results_to_word(csv_path)
            append_company_info_and_disclaimer(doc_path)
            return upload_to_container(doc_path)
    except Exception:
        return None


def describe_pages(pdf_path):
    if image_paths := pdf_to_images(pdf_path):
        if any(image_paths):
            resize_images(image_paths, 1280, False)
            llm_results = image_chats(["qwen25_vl_7b_hyperbolic", "qwen25_vl_7b_dashscope", "qwen_vl_plus_dashscope"], get_prompt("is_plain_text"), image_paths)
            image_paths = [image_path if image_path and llm_results[image_path] and ("False" in llm_results[image_path] or "false" in llm_results[image_path]) else None for image_path in image_paths]
            if any(image_paths):
                llm_results = image_chats(["qwen25_vl_72b_hyperbolic", "qwen25_vl_72b_dashscope", "qwen_vl_max_dashscope"], get_prompt("describe_visual_elements"), image_paths)
                return [llm_results[image_path] if image_path and llm_results[image_path] and len(llm_results[image_path]) > 50 else None for image_path in image_paths]


def extract_page_texts(pdf_path, pages_per_chunk, overlap_length=100):
    for attempt, config in enumerate(random.sample(DOCUMENT_CONFIGS, len(DOCUMENT_CONFIGS)), 1):
        try:
            client = DocumentAnalysisClient(endpoint=config["endpoint"], credential=AzureKeyCredential(config["api_key"]))
            with open(pdf_path, "rb") as f:
                page_texts = ["\n".join([line.content for line in page.lines]) for page in client.begin_analyze_document("prebuilt-read", f).result().pages]
                for i in range(len(page_texts)):
                    if i > 0 and i % pages_per_chunk == 0:
                        last_chunk = page_texts[i - 1]
                        overlap_text = last_chunk[-overlap_length:] if len(last_chunk) > overlap_length else ""
                        page_texts[i] = overlap_text + " " + page_texts[i]
                return page_texts
        except Exception:
            continue
    try:
        with fitz.open(pdf_path) as f:
            page_texts = [page.get_text() for page in f]
            for i in range(len(page_texts)):
                if i > 0 and i % pages_per_chunk == 0:
                    last_chunk = page_texts[i - 1]
                    overlap_text = last_chunk[-overlap_length:] if len(last_chunk) > overlap_length else ""
                    page_texts[i] = overlap_text + " " + page_texts[i]
            return page_texts
    except Exception as e:
        print(f"Failed to extract page texts: {e}")
        return None


def parse_pdfs(pdf_paths, pdf_type=None):
    pdf_contents = {}
    if pdf_type == "Plain Text":
        pages_per_chunk = 4
        overlap_length = 1000
        for pdf_path in pdf_paths:
            if page_texts := extract_page_texts(pdf_path, pages_per_chunk, overlap_length):
                page_contents = {i // pages_per_chunk + 1: "\n\n".join(page_text for page_text in page_texts[i:i + pages_per_chunk]) for i in range(0, len(page_texts), pages_per_chunk)}
                pdf_contents.update({(len(pdf_contents) + key): value for key, value in page_contents.items()})
    elif pdf_type == "With Visuals":
        pages_per_chunk = 2
        overlap_length = 500
        for pdf_path in pdf_paths:
            requests = [
                (describe_pages, (pdf_path,)),
                (extract_page_texts, (pdf_path, pages_per_chunk, overlap_length))
            ]
            results = {function.__name__: result for result, function, arguments in manage_futures(requests)}
            if page_texts := results.get("extract_page_texts"):
                if page_descriptions := results.get("describe_pages"):
                    page_contents = {i // pages_per_chunk + 1: "\n\n".join((page_descriptions[i + j] + "\n\n" if i + j < len(page_descriptions) and page_descriptions[i + j] else "") + page_text for j, page_text in enumerate(page_texts[i:i + pages_per_chunk])) for i in range(0, len(page_texts), pages_per_chunk)}
                else:
                    page_contents = {i // pages_per_chunk + 1: "\n\n".join(page_text for page_text in page_texts[i:i + pages_per_chunk]) for i in range(0, len(page_texts), pages_per_chunk)}
                pdf_contents.update({(len(pdf_contents) + key): value for key, value in page_contents.items()})
    if pdf_contents:
        txt_path = os.path.join("temp-data", f"{now_and_choices()}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(str(pdf_contents))
        return pdf_contents, txt_path
    return {}, ""


def parse_csvs(csv_paths):
    contents = {}
    for i, csv_path in enumerate(csv_paths):
        records = pd.read_csv(csv_path, encoding="utf-8").to_dict("records")
        contents[i] = records
    return json.dumps(contents, indent=4, ensure_ascii=False)


def process_csv_content(csv_path, column_to_process, column_of_results, system_message):
    df = pd.read_csv(csv_path, encoding="utf-8")
    df[column_of_results] = None
    valid_mask = df[column_to_process].notna()
    contents = df.loc[valid_mask, column_to_process].tolist()
    user_messages = [f"<{column_to_process}>\n{content}\n</{column_to_process}>" for content in contents]
    results = text_chats(["deepseek_v3_openrouter", "deepseek_v3_hyperbolic"], system_message, user_messages)
    df.loc[valid_mask, column_of_results] = [results[i] for i in range(1, len(contents) + 1)]
    csv_path = csv_path.replace(".csv", "_processed.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8")
    return csv_path


def read_txt(txt_path):
    with open(txt_path, "r", encoding="utf-8") as f:
        return ast.literal_eval(f.read())


def azure_tts(text, voice_gender):
    wav_path = os.path.join("temp-data", f"{now_and_choices()}.wav")
    for attempt, config in enumerate(random.sample(SPEECH_CONFIGS, len(SPEECH_CONFIGS)), 1):
        try:
            speech_config = speechsdk.SpeechConfig(config["api_key"], config["region"])
            speech_config.speech_synthesis_voice_name = "zh-CN-YunxiaoMultilingualNeural" if voice_gender == "male" else "zh-CN-XiaoyuMultilingualNeural"
            synthesizer = speechsdk.SpeechSynthesizer(speech_config, speechsdk.AudioConfig(filename=wav_path))
            if synthesizer.speak_text_async(text).get():
                return wav_path
        except Exception as e:
            print(f"Azure TTS attempt {attempt} failed: {e}")
    return None


def generate_audios_serially(texts, voice_gender):
    return {key: azure_tts(text, voice_gender) for key, text in texts.items()}


def merge_audios(audios):
    valid_audios = []
    for key in sorted(audios):
        try:
            with wave.open(audios[key], "rb") as f:
                if not valid_audios:
                    params = f.getparams()
                valid_audios.append(audios[key])
        except Exception:
            continue
    if valid_audios:
        wav_path = os.path.join("temp-data", f"{now_and_choices()}.wav")
        with wave.open(wav_path, "wb") as out_f:
            out_f.setparams(params)
            for valid_audio in valid_audios:
                with wave.open(valid_audio, "rb") as in_f:
                    out_f.writeframes(in_f.readframes(in_f.getnframes()))
        return wav_path


def outline_for_audio_interpretation(csv_path, user_requirements, doc_content):
    system_message = get_prompt("outline_for_audio_interpretation")
    user_message = f"<user_requirements>\n{user_requirements}\n</user_requirements>\n<doc_content>\n{doc_content}\n</doc_content>"
    outline = extract_text_with_xml_tag(text_chat(["minimax_01", "minimax_01_openrouter"], system_message, user_message), "outline")
    headings_and_notes = extract_text_with_pattern(outline, r"Dialogue Question of Chapter \d+:|Notes:")
    if isinstance(headings_and_notes, dict):
        *headings, notes = headings_and_notes.items()
        df = pd.DataFrame(columns=list(dict(headings).values()))
        if os.path.isfile(csv_path):
            df = pd.concat([pd.read_csv(csv_path, encoding="utf-8"), df], axis=1)
        df.to_csv(csv_path, index=False, encoding="utf-8")
        return outline, None if notes[1] == "None" else notes[1]
    return None


def information_for_audio_interpretation(doc_contents, outline):
    system_message = get_prompt("information_for_audio_interpretation")
    user_messages = [f"<outline>\n{outline}\n</outline>\n<doc_content>\n{doc_content}\n</doc_content>" for doc_content in doc_contents.values()]
    llm_results = text_chats(["mistral_small_31_openrouter"], system_message, user_messages)
    return {i: extract_text_with_pattern(extract_text_with_xml_tag(llm_results[i], "information_for_questions"), r"Information for Question \d+:") for i in llm_results}


def information_for_questions_to_csv(csv_path, information_for_questions):
    df = pd.read_csv(csv_path, encoding="utf-8")
    start_column = next((i for i, column in enumerate(df.columns) if df[column].isna().all()), 0)
    df = pd.concat([df, pd.DataFrame([[None] * start_column + [None if information_for_question[key] == "None" else information_for_question[key] for key in range(1, len(information_for_question) + 1)] for key, information_for_question in sorted(information_for_questions.items())], columns=df.columns)])
    df.to_csv(csv_path, index=False, encoding="utf-8")


def scripts_for_audio_interpretation(csv_path, notes):
    system_message = get_prompt("script_for_audio_interpretation")
    df = pd.read_csv(csv_path, encoding="utf-8")
    default_notes = "Focus on pivotal findings and their business relevance. Provide explanations for technical terms if any. Decode complex ideas using everyday examples and comparisons. Conclude with key takeaways and a final statement."
    user_messages = [f"<question>\n{column}\n</question>\n<information_for_question>\n{information_for_question}\n</information_for_question>\n<notes>\n{notes if notes else default_notes}\n</notes>" for column in df.columns if len(information_for_question := "\n\n".join(df[column].dropna().astype(str))) >= 300]
    llm_results = text_chats(["mistral_small_31_openrouter"], system_message, user_messages)
    return {i: f"{df.columns[i-1]}\n{filter_words(extract_text_with_xml_tag(llm_results[i], 'script'), ['*', '首先，', '其次，', '再次，', '最后，', '然而，', '然而', '此外，', '此外', '除此之外，', '总之，', '总而言之，', '总的来说，', '综上所述，', '中共', '中国共产党'])}" for i in llm_results}


def generate_audio_interpretation(txt_path, user_requirements, voice_gender, to_email=None):
    csv_path = os.path.join("temp-data", f"{now_and_choices()}.csv")
    doc_contents = read_txt(txt_path)
    for attempt in range(3):
        try:
            outline, notes = outline_for_audio_interpretation(csv_path, user_requirements, "\n\n".join(doc_contents[key] for key in sorted(doc_contents)))
            if outline and csv_path:
                for attempt in range(3):
                    try:
                        information_for_questions = information_for_audio_interpretation(doc_contents, outline)
                        if information_for_questions:
                            information_for_questions_to_csv(csv_path, information_for_questions)
                            for attempt in range(3):
                                try:
                                    scripts = scripts_for_audio_interpretation(csv_path, notes)
                                    if scripts:
                                        for attempt in range(3):
                                            try:
                                                audios = generate_audios_serially(scripts, voice_gender)
                                                if audios and sum(1 for audio in audios.values() if not audio) / len(audios) < 0.2:
                                                    for attempt in range(3):
                                                        try:
                                                            wav_path = merge_audios(audios)
                                                            if wav_path:
                                                                return upload_and_email_file(wav_path, to_email)
                                                        except Exception:
                                                            print(f"Step 5 attempt {attempt + 1} failed")
                                                    return None
                                            except Exception:
                                                print(f"Step 4 attempt {attempt + 1} failed")
                                        return None
                                except Exception:
                                    print(f"Step 3 attempt {attempt + 1} failed")
                            return None
                    except Exception:
                        print(f"Step 2 attempt {attempt + 1} failed")
                return None
        except Exception:
            print(f"Step 1 attempt {attempt + 1} failed")
    return None


def calculator(expression):
    getcontext().prec = 50
    try:
        return eval(expression, {"__builtins__": None, "math": math, "Decimal": Decimal})
    except Exception:
        return None
