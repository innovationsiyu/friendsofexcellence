from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
import azure.cognitiveservices.speech as speechsdk
from ast import literal_eval
import os
import requests
import time
import importlib
import base64
import random
import pandas as pd
import json
from pathvalidate import sanitize_filename
from PIL import Image
import fitz
import re
import wave
import math
from decimal import Decimal, getcontext
from scraper import get_web_texts, get_img_lists
from request_web_search import reader_search, tavily_answer, tavily_news_by_freshness, bing_by_freshness, bing_by_year_range, duckduckgo_by_freshness, duckduckgo_by_year_range, google_scholar_by_year_range, google_patents_by_year_range, google_by_freshness, google_answer, exa_by_freshness, exa_by_year_range, exa_news_by_freshness, exa_paper_by_freshness, exa_paper_by_year_range
from aife_time import now_in_filename, get_recent_dates_iso
from aife_utils import retrieve, manage_thread, upload_to_container, upload_and_send

OPENROUTER_API_KEY = retrieve("OpenRouter")
RAINBOWEYE_API_KEY = retrieve("RainbowEye")
DEEPINFRA_API_KEY = retrieve("DeepInfra")
DASHSCOPE_API_KEY = retrieve("DashScope")
SILICONFLOW_API_KEY = retrieve("SiliconFlow")
PERPLEXITY_API_KEY = retrieve("Perplexity")
DEEPSEEK_API_KEY = retrieve("DeepSeek")
MISTRAL_API_KEY = retrieve("Mistral")
YI_API_KEY = retrieve("Lingyiwanwu")
MINIMAX_API_KEY = retrieve("MiniMax")
XAI_API_KEY = retrieve("xAI")
XAI2_API_KEY = retrieve("xAI2")
XAI3_API_KEY = retrieve("xAI3")
EXCELLENCE_API_KEY = retrieve("ExcellenceKey")
EXCELLENCE_ENDPOINT = retrieve("ExcellenceEndpoint")
EXCELLENCE2_API_KEY = retrieve("Excellence2Key")
EXCELLENCE2_ENDPOINT = retrieve("Excellence2Endpoint")

DOCUMENT_CONFIGS = [{"api_key": retrieve("YusiMultiKey"), "endpoint": retrieve("YusiMultiEndpoint")},
                   {"api_key": retrieve("YusiMulti2Key"), "endpoint": retrieve("YusiMulti2Endpoint")}]

SPEECH_CONFIGS = [{"api_key": retrieve("SpeechKey"), "region": "eastus"},
                  {"api_key": retrieve("Speech2Key"), "region": "westus2"}]


def execute(tool_calls):
    try:
        results = {
            f"{name}({arguments})": globals().get(name)(**literal_eval(arguments))
            for tool_call in tool_calls
            if (function := tool_call.get("function"))
            if (name := function.get("name")) and (arguments := function.get("arguments"))
            if name in globals()
        }
        return results
    except Exception as e:
        print(f"Failed to execute tool calls: {e}")
        return None


def request_llm(url, headers, data, delay=1):
    for attempt in range(3):
        try:
            print(f"Sending request to {url} with {data}")
            response = requests.post(url, headers=headers, json=data, timeout=180).json()
            print(response)
            if (message := response.get("choices", [{}])[0].get("message", {})):
                if (tool_calls := message.get("tool_calls")):
                    if (results := execute(tool_calls)):
                        return f"The following dictionary contains the results:\n{results}"
                elif (content := message.get("content")):
                    return content
            raise Exception("Invalid response structure or execution failed")
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < 2:
                time.sleep(delay)
                delay *= 2
    print("Failed to get a valid response after maximum retries")
    return None


class LLM:
    def __init__(self, url, api_key):
        self.url = url
        self.api_keys = [api_key] if isinstance(api_key, str) else api_key

    def __call__(self, messages, model, temperature, top_p, response_format=None, tools=None):
        headers = {
            "Authorization": f"Bearer {random.choice(self.api_keys)}"
        }
        data = {
            "messages": messages,
            "model": model,
            "temperature": temperature,
            "top_p": top_p,
            **({"response_format": response_format} if response_format else {}),
            **({"tools": tools} if tools else {})
        }
        return request_llm(self.url, headers, data)


class Azure:
    def __init__(self, endpoint, api_key):
        self.endpoint = endpoint
        self.api_key = api_key

    def __call__(self, messages, model, temperature, top_p, response_format=None, tools=None):
        url = f"{self.endpoint}openai/deployments/{model}/chat/completions?api-version=2024-05-01-preview"
        headers = {
            "api-key": self.api_key
        }
        data = {
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            **({"response_format": response_format} if response_format else {}),
            **({"tools": tools} if tools else {})
        }
        return request_llm(url, headers, data)


openrouter = LLM("https://openrouter.ai/api/v1/chat/completions", OPENROUTER_API_KEY)
rainboweye = LLM("https://gitaigc.com/v1/chat/completions", RAINBOWEYE_API_KEY)
deepinfra = LLM("https://api.deepinfra.com/v1/openai/chat/completions", DEEPINFRA_API_KEY)
dashscope = LLM("https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions", DASHSCOPE_API_KEY)
siliconflow = LLM("https://api.siliconflow.cn/v1/chat/completions", SILICONFLOW_API_KEY)
perplexity = LLM("https://api.perplexity.ai/chat/completions", PERPLEXITY_API_KEY)
deepseek = LLM("https://api.deepseek.com/chat/completions", DEEPSEEK_API_KEY)
mistral = LLM("https://api.mistral.ai/v1/chat/completions", MISTRAL_API_KEY)
lingyiwanwu = LLM("https://api.lingyiwanwu.com/v1/chat/completions", YI_API_KEY)
minimax = LLM("https://api.minimax.chat/v1/text/chatcompletion_v2", MINIMAX_API_KEY)
xai = LLM("https://api.x.ai/v1/chat/completions", [XAI_API_KEY, XAI2_API_KEY, XAI3_API_KEY])
excellence = Azure(EXCELLENCE_ENDPOINT, EXCELLENCE_API_KEY)
excellence2 = Azure(EXCELLENCE2_ENDPOINT, EXCELLENCE2_API_KEY)


def get_prompt(prompt, **arguments):
    if arguments:
        return getattr(importlib.import_module(f"aife_prompts.{prompt}"), prompt).format(**arguments)
    else:
        return getattr(importlib.import_module(f"aife_prompts.{prompt}"), prompt)


def get_response_format(response_format):
    if response_format:
        return getattr(importlib.import_module(f"aife_response_formats.{response_format}"), response_format)
    return None


def get_tools(tools):
    if tools:
        return [getattr(importlib.import_module("aife_tools"), tool) for tool in tools]
    return None


class Chat:
    def __call__(self, llms, messages, response_format=None, tools=None):
        for llm in llms:
            try:
                results = globals()[llm_dict[llm]["name"]](messages, **llm_dict[llm]["arguments"], response_format=response_format, tools=tools)
                if results:
                    return results
            except Exception:
                continue
        return None

chat = Chat()

def internal_text_chat(ai, system_message, user_message, response_format=None, tools=None):
    llms = ai_dict[ai]["llms"]
    messages = [{"role": "system", "content": system_message}, {"role": "user", "content": user_message}]
    return chat(llms, messages, response_format, tools)

def internal_image_chat(ai, user_message, image_path):
    llms = ai_dict[ai]["llms"]
    messages = [{"role": "user", "content": [{"type": "text", "text": user_message}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64.b64encode(open(image_path, 'rb').read()).decode('utf-8')}"}}]}]
    return chat(llms, messages)


ai_dict = {
    "Perplexity for quick web search and analysis": {
        "category": "chat_only",
        "llms": ["perplexity_openrouter", "perplexity"],
        "system_message": "quick_web_search_and_analysis",
        "response_format": None,
        "tools": None,
        "backend_ais": None,
        "max_length": 125000,
        "intro": "Perplexity快速搜索公开信息并回答问题"
    },
    "GPT for web search and analysis": {
        "category": "function_calling",
        "llms": ["gpt4o_openrouter", "gpt4o_rainboweye", "gpt4o_excellence"],
        "system_message": "web_search_and_analysis",
        "response_format": None,
        "tools": ["basic_search_func", "broader_search_func", "get_web_texts_func"],
        "backend_ais": None,
        "max_length": 125000,
        "intro": "Google Tavily Jina搜索公开信息 GPT回答问题"
    },
    "GPT for serious web search and analysis": {
        "category": "function_calling",
        "llms": ["gpt4o_openrouter", "gpt4o_rainboweye", "gpt4o_excellence"],
        "system_message": "serious_web_search_and_analysis",
        "response_format": None,
        "tools": ["serious_search_by_freshness_func", "serious_search_by_year_range_func", "news_search_by_freshness_func", "scholar_search_by_freshness_func", "scholar_search_by_year_range_func", "patents_search_by_year_range_func", "get_web_texts_func"],
        "backend_ais": None,
        "max_length": 125000,
        "intro": "Google、Bing、Duckduckgo、Exa、Tavily严谨搜索公开信息 GPT回答问题"
    },
    "GPT Qwen for dense visual reading": {
        "category": "dense_visual",
        "llms": ["gpt4o_openrouter", "gpt4o_rainboweye", "gpt4o_excellence"],
        "system_message": "interpret_documents_with_tools",
        "response_format": None,
        "tools": ["generate_audio_interpretation_func", "generate_audio_storytelling_func"],
        "backend_ais": ["qwen2_vl_72b_siliconflow", "qwen2_vl_72b_openrouter", "qwen_vl_max_dashscope"],
        "max_length": 125000,
        "intro": "GPT和Qwen解读视觉元素为主的文档 3万字符以内 还可生成音频解读"
    },
    "GPT Pixtral for dense visual reading": {
        "category": "dense_visual",
        "llms": ["gpt4o_openrouter", "gpt4o_rainboweye", "gpt4o_excellence"],
        "system_message": "interpret_documents_with_tools",
        "response_format": None,
        "tools": ["generate_audio_interpretation_func", "generate_audio_storytelling_func"],
        "backend_ais": ["pixtral_large", "pixtral_large_openrouter"],
        "max_length": 125000,
        "intro": "GPT和Pixtral解读视觉元素为主的文档 3万字符以内 还可生成音频解读"
    },
    "GPT Qwen for blended layout reading": {
        "category": "blended_layout",
        "llms": ["gpt4o_openrouter", "gpt4o_rainboweye", "gpt4o_excellence"],
        "system_message": "interpret_documents_with_tools",
        "response_format": None,
        "tools": ["generate_audio_interpretation_func", "generate_audio_storytelling_func"],
        "backend_ais": ["qwen2_vl_72b_siliconflow", "qwen2_vl_72b_openrouter", "qwen_vl_max_dashscope"],
        "max_length": 125000,
        "intro": "GPT和Qwen解读视觉与文字并重的文档 10万字符以内 还可生成音频解读"
    },
    "GPT Pixtral for blended layout reading": {
        "category": "blended_layout",
        "llms": ["gpt4o_openrouter", "gpt4o_rainboweye", "gpt4o_excellence"],
        "system_message": "interpret_documents_with_tools",
        "response_format": None,
        "tools": ["generate_audio_interpretation_func", "generate_audio_storytelling_func"],
        "backend_ais": ["pixtral_large", "pixtral_large_openrouter"],
        "max_length": 125000,
        "intro": "GPT和Pixtral解读视觉与文字并重的文档 10万字符以内 还可生成音频解读"
    },
    "GPT for plain text reading": {
        "category": "plain_text",
        "llms": ["gpt4o_openrouter", "gpt4o_rainboweye", "gpt4o_excellence"],
        "system_message": "interpret_documents_with_tools",
        "response_format": None,
        "tools": ["generate_audio_interpretation_func", "generate_audio_storytelling_func"],
        "backend_ais": None,
        "max_length": 125000,
        "intro": "GPT解读纯文字文档 10万字符以内 还可生成音频解读"
    },
    "Gemini for long plain text reading": {
        "category": "long_plain_text",
        "llms": ["gemini15_flash_8b_openrouter"],
        "system_message": "interpret_documents",
        "response_format": None,
        "tools": None,
        "backend_ais": None,
        "max_length": 998000,
        "intro": "Gemini解读纯文字长文档 接近100万字符"
    },
    "Qwen for image chat": {
        "category": "vision",
        "llms": ["qwen2_vl_72b_siliconflow", "qwen2_vl_72b_openrouter", "qwen_vl_max_dashscope"],
        "backend_ais": None,
        "max_length": 3000,
        "intro": "阿里云通义千问最新视觉模型"
    },
    "Pixtral for image chat": {
        "category": "vision",
        "llms": ["pixtral_large", "pixtral_large_openrouter"],
        "backend_ais": None,
        "max_length": 80000,
        "intro": "Mistral AI旗舰视觉模型"
    },
    "Qwen for Chinese and English translation": {
        "category": "chat_only",
        "llms": ["qwen25_72b_deepinfra", "qwen25_72b_openrouter", "qwen25_72b_dashscope", "qwen25_72b_siliconflow"],
        "system_message": "en_zh_translation_and_checking",
        "response_format": None,
        "tools": None,
        "backend_ais": None,
        "max_length": 31000,
        "intro": "阿里云通义千问旗舰模型执行中英互译相关任务"
    },
    "Mistral for wider language translation": {
        "category": "chat_only",
        "llms": ["mistral_large", "mistral_large_openrouter"],
        "system_message": "wider_language_translation_and_checking",
        "response_format": None,
        "tools": None,
        "backend_ais": None,
        "max_length": 125000,
        "intro": "Mistral AI旗舰模型执行更多语种翻译相关任务"
    },
    "GPT for maths questions": {
        "category": "chat_only",
        "llms": ["gpt4o_openrouter"],
        "system_message": "maths_research_and_learning",
        "response_format": None,
        "tools": ["calculator_func"],
        "backend_ais": None,
        "max_length": 125000,
        "intro": "GPT讲解数学问题 可以使用计算器"
    },
    "o1-mini for critical thinking": {
        "category": "chat_only",
        "llms": ["o1_mini_openrouter"],
        "system_message": "logic_checking_and_critical_thinking",
        "response_format": None,
        "tools": None,
        "backend_ais": None,
        "max_length": 125000,
        "intro": "o1-mini辅助批判性思考"
    },
    "GPT for text chat": {
        "category": "chat_only",
        "llms": ["gpt4o_openrouter", "gpt4o_rainboweye", "gpt4o_excellence"],
        "system_message": "chat_only",
        "response_format": None,
        "tools": None,
        "backend_ais": None,
        "max_length": 125000,
        "intro": "OpenAI: GPT-4o"
    },
    "Claude for text chat": {
        "category": "chat_only",
        "llms": ["claude35_sonnet_openrouter"],
        "system_message": "chat_only",
        "response_format": None,
        "tools": None,
        "backend_ais": None,
        "max_length": 199000,
        "intro": "Anthropic: Claude 3.5 Sonnet"
    },
    "Grok for text chat": {
        "category": "chat_only",
        "llms": ["grok2_xai", "grok2_openrouter"],
        "system_message": "chat_only",
        "response_format": None,
        "tools": None,
        "backend_ais": None,
        "max_length": 130000,
        "intro": "马斯克xAI全球最大超算集群训练的模型"
    },
    "Llama for text chat": {
        "category": "chat_only",
        "llms": ["llama33_70b_deepinfra", "llama33_70b_openrouter"],
        "system_message": "chat_only",
        "response_format": None,
        "tools": None,
        "backend_ais": None,
        "max_length": 130000,
        "intro": "Meta: Llama 3.3 70B Instruct"
    },
    "Yi for text chat": {
        "category": "chat_only",
        "llms": ["yi_lightning"],
        "system_message": "chat_only",
        "response_format": None,
        "tools": None,
        "backend_ais": None,
        "max_length": 15000,
        "intro": "李开复零一万物旗舰模型"
    },
    "Qwen for text chat": {
        "category": "chat_only",
        "llms": ["qwen25_72b_deepinfra", "qwen25_72b_openrouter", "qwen25_72b_dashscope", "qwen25_72b_siliconflow"],
        "system_message": "chat_only",
        "response_format": None,
        "tools": None,
        "backend_ais": None,
        "max_length": 31000,
        "intro": "阿里云通义千问旗舰模型"
    },
    "MiniMax for text chat": {
        "category": "chat_only",
        "llms": ["abab7"],
        "system_message": "chat_only",
        "response_format": None,
        "tools": None,
        "backend_ais": None,
        "max_length": 244000,
        "intro": "MiniMax最新聊天模型"
    },
    "DeepSeek for text chat": {
        "category": "chat_only",
        "llms": ["deepseek", "deepseek_openrouter", "deepseek_siliconflow"],
        "system_message": "chat_only",
        "response_format": None,
        "tools": None,
        "backend_ais": None,
        "max_length": 64500,
        "intro": "DeepSeek最新聊天模型"
    },
    "Nemotron for text chat": {
        "category": "chat_only",
        "llms": ["nemotron_70b_deepinfra", "nemotron_70b_openrouter"],
        "system_message": "chat_only",
        "response_format": None,
        "tools": None,
        "backend_ais": None,
        "max_length": 15000,
        "intro": "英伟达利用合成数据训练的模型"
    },
    "LFM for text chat": {
        "category": "chat_only",
        "llms": ["lfm_40b_openrouter"],
        "system_message": "chat_only",
        "response_format": None,
        "tools": None,
        "backend_ais": None,
        "max_length": 65000,
        "intro": "Liquid AI的非Transformer架构模型"
    },
    "gemini15_flash_8b_openrouter": {
        "category": "internal_text_chat",
        "llms": ["gemini15_flash_8b_openrouter"],
        "system_message": None,
        "response_format": None,
        "tools": None,
        "backend_ais": None,
        "max_length": None,
        "intro": "internal_text_chat"
    },
    "gpt4o_openrouter": {
        "category": "internal_text_chat",
        "llms": ["gpt4o_openrouter", "gpt4o_rainboweye"],
        "system_message": None,
        "response_format": None,
        "tools": None,
        "backend_ais": None,
        "max_length": None,
        "intro": "internal_text_chat"
    },
    "gpt4o_rainboweye": {
        "category": "internal_text_chat",
        "llms": ["gpt4o_rainboweye", "gpt4o_openrouter"],
        "system_message": None,
        "response_format": None,
        "tools": None,
        "backend_ais": None,
        "max_length": None,
        "intro": "internal_text_chat"
    },
    "gpt4o_mini_openrouter": {
        "category": "internal_text_chat",
        "llms": ["gpt4o_mini_openrouter", "gpt4o_mini_rainboweye", "gpt4o_mini_excellence2"],
        "system_message": None,
        "response_format": None,
        "tools": None,
        "backend_ais": None,
        "max_length": None,
        "intro": "internal_text_chat"
    },
    "gpt4o_mini_excellence2": {
        "category": "internal_text_chat",
        "llms": ["gpt4o_mini_rainboweye", "gpt4o_mini_openrouter", "gpt4o_mini_excellence2"],
        "system_message": None,
        "response_format": None,
        "tools": None,
        "backend_ais": None,
        "max_length": None,
        "intro": "internal_text_chat"
    },
    "gpt4o_mini_rainboweye": {
        "category": "internal_text_chat",
        "llms": ["gpt4o_mini_excellence2", "gpt4o_mini_openrouter", "gpt4o_mini_rainboweye"],
        "system_message": None,
        "response_format": None,
        "tools": None,
        "backend_ais": None,
        "max_length": None,
        "intro": "internal_text_chat"
    },
    "pixtral_large": {
        "category": "internal_image_chat",
        "llms": ["pixtral_large", "pixtral_large_openrouter"],
        "backend_ais": None,
        "max_length": None,
        "intro": "internal_image_chat"
    },
    "pixtral_large_openrouter": {
        "category": "internal_image_chat",
        "llms": ["pixtral_large_openrouter", "pixtral_large"],
        "backend_ais": None,
        "max_length": None,
        "intro": "internal_image_chat"
    },
    "qwen2_vl_72b_siliconflow": {
        "category": "internal_image_chat",
        "llms": ["qwen2_vl_72b_siliconflow", "qwen2_vl_72b_openrouter", "qwen_vl_max_dashscope"],
        "backend_ais": None,
        "max_length": None,
        "intro": "internal_image_chat"
    },
    "qwen2_vl_72b_openrouter": {
        "category": "internal_image_chat",
        "llms": ["qwen2_vl_72b_openrouter", "qwen2_vl_72b_siliconflow", "qwen_vl_max_dashscope"],
        "backend_ais": None,
        "max_length": None,
        "intro": "internal_image_chat"
    },
    "qwen_vl_max_dashscope": {
        "category": "internal_image_chat",
        "llms": ["qwen_vl_max_dashscope", "qwen2_vl_72b_siliconflow", "qwen2_vl_72b_openrouter"],
        "backend_ais": None,
        "max_length": None,
        "intro": "internal_image_chat"
    },
    "qwen2_vl_7b_siliconflow": {
        "category": "internal_image_chat",
        "llms": ["qwen2_vl_7b_siliconflow", "qwen2_vl_7b_openrouter", "qwen_vl_plus_dashscope"],
        "backend_ais": None,
        "max_length": None,
        "intro": "internal_image_chat"
    },
    "qwen2_vl_7b_openrouter": {
        "category": "internal_image_chat",
        "llms": ["qwen2_vl_7b_openrouter", "qwen2_vl_7b_siliconflow", "qwen_vl_plus_dashscope"],
        "backend_ais": None,
        "max_length": None,
        "intro": "internal_image_chat"
    },
    "qwen_vl_plus_dashscope": {
        "category": "internal_image_chat",
        "llms": ["qwen_vl_plus_dashscope", "qwen2_vl_7b_siliconflow", "qwen2_vl_7b_openrouter"],
        "backend_ais": None,
        "max_length": None,
        "intro": "internal_image_chat"
    }
}

llm_dict = {
    "o1_mini_openrouter": {
        "name": "openrouter",
        "arguments": {
            "model": "openai/o1-mini-2024-09-12",
            "temperature": 0.5,
            "top_p": 0.9
        }
    },
    "claude35_sonnet_openrouter": {
        "name": "openrouter",
        "arguments": {
            "model": "anthropic/claude-3.5-sonnet",
            "temperature": 0.5,
            "top_p": 0.9
        }
    },
    "grok2_xai": {
        "name": "xai",
        "arguments": {
            "model": "grok-beta",
            "temperature": 0.5,
            "top_p": 0.9
        }
    },
    "grok2_openrouter": {
        "name": "openrouter",
        "arguments": {
            "model": "x-ai/grok-beta",
            "temperature": 0.5,
            "top_p": 0.9
        }
    },
    "llama33_70b_deepinfra": {
        "name": "deepinfra",
        "arguments": {
            "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "temperature": 0.5,
            "top_p": 0.9
        }
    },
    "llama33_70b_openrouter": {
        "name": "openrouter",
        "arguments": {
            "model": "meta-llama/llama-3.3-70b-instruct",
            "temperature": 0.5,
            "top_p": 0.9
        }
    },
    "perplexity_openrouter": {
        "name": "openrouter",
        "arguments": {
            "model": "perplexity/llama-3.1-sonar-large-128k-online",
            "temperature": 0.4,
            "top_p": 0.9
        }
    },
    "perplexity": {
        "name": "perplexity",
        "arguments": {
            "model": "llama-3.1-sonar-large-128k-online",
            "temperature": 0.4,
            "top_p": 0.9
        }
    },
    "gemini15_flash_8b_openrouter": {
        "name": "openrouter",
        "arguments": {
            "model": "google/gemini-flash-1.5-8b",
            "temperature": 0.4,
            "top_p": 0.9
        }
    },
    "mistral_large": {
        "name": "mistral",
        "arguments": {
            "model": "mistral-large-latest",
            "temperature": 0.5,
            "top_p": 0.9
        }
    },
    "mistral_large_openrouter": {
        "name": "openrouter",
        "arguments": {
            "model": "mistralai/mistral-large",
            "temperature": 0.5,
            "top_p": 0.9
        }
    },
    "pixtral_large": {
        "name": "mistral",
        "arguments": {
            "model": "pixtral-large-latest",
            "temperature": 0.3,
            "top_p": 0.9
        }
    },
    "pixtral_large_openrouter": {
        "name": "openrouter",
        "arguments": {
            "model": "mistralai/pixtral-large-2411",
            "temperature": 0.3,
            "top_p": 0.9
        }
    },
    "yi_lightning": {
        "name": "lingyiwanwu",
        "arguments": {
            "model": "yi-lightning",
            "temperature": 0.5,
            "top_p": 0.9
        }
    },
    "deepseek_siliconflow": {
        "name": "siliconflow",
        "arguments": {
            "model": "deepseek-ai/DeepSeek-V2.5",
            "temperature": 0.3,
            "top_p": 0.9
        }
    },
    "deepseek_openrouter": {
        "name": "openrouter",
        "arguments": {
            "model": "deepseek/deepseek-chat",
            "temperature": 0.3,
            "top_p": 0.9
        }
    },
    "deepseek": {
        "name": "deepseek",
        "arguments": {
            "model": "deepseek-chat",
            "temperature": 0.3,
            "top_p": 0.9
        }
    },
    "abab7": {
        "name": "minimax",
        "arguments": {
            "model": "abab7-chat-preview",
            "temperature": 0.5,
            "top_p": 0.9
        }
    },
    "nemotron_70b_deepinfra": {
        "name": "deepinfra",
        "arguments": {
            "model": "nvidia/Llama-3.1-Nemotron-70B-Instruct",
            "temperature": 0.5,
            "top_p": 0.9
        }
    },
    "nemotron_70b_openrouter": {
        "name": "openrouter",
        "arguments": {
            "model": "nvidia/llama-3.1-nemotron-70b-instruct",
            "temperature": 0.5,
            "top_p": 0.9
        }
    },
    "lfm_40b_openrouter": {
        "name": "openrouter",
        "arguments": {
            "model": "liquid/lfm-40b",
            "temperature": 0.5,
            "top_p": 0.9
        }
    },
    "qwen25_72b_deepinfra": {
        "name": "deepinfra",
        "arguments": {
            "model": "Qwen/Qwen2.5-72B-Instruct",
            "temperature": 0.5,
            "top_p": 0.9
        }
    },
    "qwen25_72b_openrouter": {
        "name": "openrouter",
        "arguments": {
            "model": "qwen/qwen-2.5-72b-instruct",
            "temperature": 0.5,
            "top_p": 0.9
        }
    },
    "qwen25_72b_dashscope": {
        "name": "dashscope",
        "arguments": {
            "model": "qwen2.5-72b-instruct",
            "temperature": 0.5,
            "top_p": 0.9
        }
    },
    "qwen25_72b_siliconflow": {
        "name": "siliconflow",
        "arguments": {
            "model": "Qwen/Qwen2.5-72B-Instruct",
            "temperature": 0.5,
            "top_p": 0.9
        }
    },
    "qwen_vl_max_dashscope": {
        "name": "dashscope",
        "arguments": {
            "model": "qwen-vl-max-0809",
            "temperature": 0.5,
            "top_p": 0.9
        }
    },
    "qwen2_vl_72b_siliconflow": {
        "name": "siliconflow",
        "arguments": {
            "model": "Qwen/Qwen2-VL-72B-Instruct",
            "temperature": 0.5,
            "top_p": 0.9
        }
    },
    "qwen2_vl_72b_openrouter": {
        "name": "openrouter",
        "arguments": {
            "model": "qwen/qwen-2-vl-72b-instruct",
            "temperature": 0.5,
            "top_p": 0.9
        }
    },
    "qwen_vl_plus_dashscope": {
        "name": "dashscope",
        "arguments": {
            "model": "qwen-vl-plus-0809",
            "temperature": 0.3,
            "top_p": 0.9
        }
    },
    "qwen2_vl_7b_siliconflow": {
        "name": "siliconflow",
        "arguments": {
            "model": "Pro/Qwen/Qwen2-VL-7B-Instruct",
            "temperature": 0.3,
            "top_p": 0.9
        }
    },
    "qwen2_vl_7b_openrouter": {
        "name": "openrouter",
        "arguments": {
            "model": "qwen/qwen-2-vl-7b-instruct",
            "temperature": 0.3,
            "top_p": 0.9
        }
    },
    "gpt4o_rainboweye": {
        "name": "rainboweye",
        "arguments": {
            "model": "gpt-4o-2024-08-06",
            "temperature": 0.5,
            "top_p": 0.9
        }
    },
    "gpt4o_openrouter": {
        "name": "openrouter",
        "arguments": {
            "model": "openai/gpt-4o-2024-08-06",
            "temperature": 0.5,
            "top_p": 0.9
        }
    },
    "gpt4o_mini_rainboweye": {
        "name": "rainboweye",
        "arguments": {
            "model": "gpt-4o-mini-2024-07-18",
            "temperature": 0.3,
            "top_p": 0.9
        }
    },
    "gpt4o_mini_openrouter": {
        "name": "openrouter",
        "arguments": {
            "model": "openai/gpt-4o-mini-2024-07-18",
            "temperature": 0.3,
            "top_p": 0.9
        }
    },
    "gpt4o_excellence": {
        "name": "excellence",
        "arguments": {
            "model": "excellence",
            "temperature": 0.5,
            "top_p": 0.9
        }
    },
    "gpt4o_mini_excellence2": {
        "name": "excellence2",
        "arguments": {
            "model": "yusi-mini",
            "temperature": 0.3,
            "top_p": 0.9
        }
    }
}

basic_search_func = {
    "type": "function",
    "function": {
        "name": "basic_search",
        "description": "Utilise Google Search to obtain the answer box, knowledge graph, and/or a list of webpages. Call this when explicit factual information is required for an expected reply.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The topic or keywords that reflect the user's intent and requirements for relevant and applicable knowledge. Consider leveraging advanced search operators, which include exact quotes (""), logical operators (AND/OR), field-specific operators, domain filters, time filters, and other search techniques to enhance precision and relevance."
                }
            },
            "required": ["query"],
            "additionalProperties": False
        }
    }
}

broader_search_func = {
    "type": "function",
    "function": {
        "name": "broader_search",
        "description": "Utilise multiple search engines to obtain more comprehensive and potentially redundant content. Call this when both explicit and implicit factual information are required for an expected reply, such as when the user requests additional related information from different perspectives or asks for further searches without specifying directions.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The topic or keywords that reflect the user's intent and requirements for relevant and applicable knowledge. Consider constructing interrogative sentences from different perspectives to broaden the search scope."
                }
            },
            "required": ["query"],
            "additionalProperties": False
        }
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
                "target_urls": {
                    "type": "array",
                    "description": "A list of URLs to simultaneously scrape text from.",
                    "items": {
                        "type": "string",
                        "description": "The URL to scrape text from."
                    }
                }
            },
            "required": ["target_urls"],
            "additionalProperties": False
        }
    }
}

news_search_by_freshness_func = {
    "type": "function",
    "function": {
        "name": "news_search_by_freshness",
        "description": "Utilise multiple search engines to access news articles, optionally filtering by freshness (number of days ago) and credible sources. Call this when details of public occurrences, particularly recent ones, are required for an expected reply, such as credible news on climate change from the past week.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The topic or keywords that reflect the user's intent and requirements for public occurrences."
                },
                "freshness": {
                    "type": ["integer", "null"],
                    "description": "The number of days to look back for filtering the results by the published date. Identify the requirement and convert it into an integer. Common values are 7 or 30. Omit this argument if there is no requirement for freshness."
                },
                "credible_only": {
                    "type": ["boolean", "null"],
                    "description": "Whether to filter results to include only those from credible sources. Set to False (default) or omit this argument if there is no requirement for credible sources."
                }
            },
            "required": ["query"],
            "additionalProperties": False
        }
    }
}

scholar_search_by_freshness_func = {
    "type": "function",
    "function": {
        "name": "scholar_search_by_freshness",
        "description": "Utilise multiple search engines to access research articles, optionally filtering by freshness (number of days ago). Call this when scholarly content, particularly recent content, is required for an expected reply, such as papers on AI ethics from the past month.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The topic or keywords that reflect the user's intent and requirements for scholarly content."
                },
                "freshness": {
                    "type": ["integer", "null"],
                    "description": "The number of days to look back for filtering the results by the published date. Identify the requirement and convert it into an integer. Common values are 7 or 30. Omit this argument if there is no requirement for freshness."
                }
            },
            "required": ["query"],
            "additionalProperties": False
        }
    }
}

serious_search_by_freshness_func = {
    "type": "function",
    "function": {
        "name": "serious_search_by_freshness",
        "description": "Utilise multiple search engines to access information in specified areas of expertise, optionally filtering by freshness (number of days ago). Call this when professional content, particularly recent content, is required for an expected reply, such as marketing strategies for humanoid robots in EU markets for the current year.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The topic or keywords that reflect the user's intent and requirements for professional content. Consider leveraging advanced search operators, which include exact quotes (""), logical operators (AND/OR), field-specific operators, domain filters, time filters, and other search techniques to enhance precision and relevance. Another suggested approach is constructing interrogative sentences from different perspectives to broaden the search scope."
                },
                "freshness": {
                    "type": ["integer", "null"],
                    "description": "The number of days to look back for filtering the results by the published date. Identify the requirement and convert it into an integer. Common values are 7 or 30. Omit this argument if there is no requirement for freshness."
                },
                "credible_only": {
                    "type": ["boolean", "null"],
                    "description": "Whether to filter results to include only those from credible sources. Set to False (default) or omit this argument if there is no requirement for credible sources."
                }
            },
            "required": ["query"],
            "additionalProperties": False
        }
    }
}

scholar_search_by_year_range_func = {
    "type": "function",
    "function": {
        "name": "scholar_search_by_year_range",
        "description": "Utilise multiple search engines to access research articles, optionally filtering by year range. Call this when scholarly content, particularly from a specific period, is required for an expected reply, such as papers on precision medicine published between 2022 and 2024.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The topic or keywords that reflect the user's intent and requirements for scholarly content."
                },
                "year_range": {
                    "type": ["array", "null"],
                    "description": "The start year and end year for filtering the results by the published date. Identify the requirement and convert it into a pair of 4-digit integers in which the first represents the start year and the second represents the end year. For example, [2022, 2024]. Omit this argument if there is no requirement for a year range.",
                    "items": {
                        "type": "integer",
                        "description": "A 4-digit integer representing a year."
                    },
                    "minItems": 2,
                    "maxItems": 2
                }
            },
            "required": ["query"],
            "additionalProperties": False
        }
    }
}

patents_search_by_year_range_func = {
    "type": "function",
    "function": {
        "name": "patents_search_by_year_range",
        "description": "Utilise Google Search to access patent files, optionally filtering by year range. Call this when patent information, particularly from a specific period, is required for an expected reply, such as patents on electric vehicles granted in the last three years.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The topic or keywords that reflect the user's intent and requirements for patent information."
                },
                "year_range": {
                    "type": ["array", "null"],
                    "description": "The start year and end year for filtering the results by the granted date. Identify the requirement and convert it into a pair of 4-digit integers in which the first represents the start year and the second represents the end year. For example, [2022, 2024]. Omit this argument if there is no requirement for a year range.",
                    "items": {
                        "type": "integer",
                        "description": "A 4-digit integer representing a year."
                    },
                    "minItems": 2,
                    "maxItems": 2
                }
            },
            "required": ["query"],
            "additionalProperties": False
        }
    }
}

serious_search_by_year_range_func = {
    "type": "function",
    "function": {
        "name": "serious_search_by_year_range",
        "description": "Utilise multiple search engines to access information in specified areas of expertise, optionally filtering by year range. Call this when professional content, particularly from a specific period, is required for an expected reply, such as the world's top tech companies by market capitalisation over the past 10 years.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The topic or keywords that reflect the user's intent and requirements for professional content. Consider leveraging advanced search operators, which include exact quotes (""), logical operators (AND/OR), field-specific operators, domain filters, time filters, and other search techniques to enhance precision and relevance. Another suggested approach is constructing interrogative sentences from different perspectives to broaden the search scope."
                },
                "year_range": {
                    "type": ["array", "null"],
                    "description": "The start year and end year for filtering the results by the published date. Identify the requirement and convert it into a pair of 4-digit integers in which the first represents the start year and the second represents the end year. For example, [2022, 2024]. Omit this argument if there is no requirement for a year range.",
                    "items": {
                        "type": "integer",
                        "description": "A 4-digit integer representing a year."
                    },
                    "minItems": 2,
                    "maxItems": 2
                },
                "credible_only": {
                    "type": ["boolean", "null"],
                    "description": "Whether to filter results to include only those from credible sources. Set to False (default) or omit this argument if there is no requirement for credible sources."
                }
            },
            "required": ["query"],
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
                    "description": "The path of the TXT file containing the entire content of the documents."
                },
                "user_requirements": {
                    "type": "string",
                    "description": "The  user's requirements for the interpretation."
                },
                "voice_gender": {
                    "type": "string",
                    "enum": ["male", "female"],
                    "description": "The user's choice of male or female voice for the audio."
                },
                "to_email": {
                    "type": ["string", "null"],
                    "description": "The user's email address for secondary delivery. This is optional, as a safeguard against potential chat session disruptions."
                },
            },
            "required": ["txt_path", "user_requirements", "voice_gender"],
            "additionalProperties": False,
        },
    },
}

generate_audio_storytelling_func = {
    "type": "function",
    "function": {
        "name": "generate_audio_storytelling",
        "description": "Generate an audio storytelling for novels or short stories. This function returns a URL for downloading the audio file.",
        "parameters": {
            "type": "object",
            "properties": {
                "txt_path": {
                    "type": "string",
                    "description": "The path of the TXT file containing the entire content of the documents."
                },
                "user_requirements": {
                    "type": "string",
                    "description": "The  user's requirements for the storytelling."
                },
                "voice_gender": {
                    "type": "string",
                    "enum": ["male", "female"],
                    "description": "The user's choice of male or female voice for the audio."
                },
                "to_email": {
                    "type": ["string", "null"],
                    "description": "The user's email address for secondary delivery. This is optional, as a safeguard against potential chat session disruptions."
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
        "description": "Evaluate a wide range of mathematical expressions with high precision (50 decimal places). Supports basic arithmetic operations and math module functions.",
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


def csv_for_search_results(query):
    csv_path = f"temp-data/{sanitize_filename(query[:20])} {now_in_filename()}.csv"
    pd.DataFrame(columns=["sequence", "title", "summary", "author", "publication_info", "url", "date", "grant_date"]).to_csv(csv_path, index=False, encoding="utf-8")
    return csv_path


def search_results_to_csv(csv_path, results, freshness=None):
    df = pd.read_csv(csv_path, encoding="utf-8")
    df = pd.concat([df, pd.DataFrame([{
        **{column: item.get(column) for column in df.columns if column in item}
    } for index, item in results.items()]).reindex(columns=df.columns)])
    df = df.drop_duplicates(subset="title", keep="first")
    if freshness:
        df = df.drop([index for index, row in df.iterrows() if row.get("date") not in get_recent_dates_iso(freshness)])
    df.to_csv(csv_path, index=False, encoding="utf-8")


def csv_for_relevant_search_results(csv_path):
    pd.DataFrame(columns=["title", "web_text", "author", "publication_info", "url", "date", "grant_date"]).to_csv(csv_path, index=False, encoding="utf-8")


def relevant_search_results_to_csv(csv_path, results):
    df = pd.read_csv(csv_path, encoding="utf-8")
    web_texts = get_web_texts([item.get("url") for index, item in results.items()])
    df = pd.concat([df, pd.DataFrame([{
        "web_text": web_texts.get(item.get("url")),
        **{column: item.get(column) for column in df.columns if column in item}
    } for index, item in results.items()]).reindex(columns=df.columns)])
    df.to_csv(csv_path, index=False, encoding="utf-8")


def csv_for_applicable_search_results(csv_path):
    pd.DataFrame(columns=["applicable_knowledge", "title", "web_text", "img_list", "author", "publication_info", "url", "date", "grant_date"]).to_csv(csv_path, index=False, encoding="utf-8")


def applicable_search_results_to_csv(csv_path, results):
    df = pd.read_csv(csv_path, encoding="utf-8")
    img_lists = get_img_lists([item.get("url") for index, item in results.items()])
    df = pd.concat([df, pd.DataFrame([{
        "img_list": img_lists.get(item.get("url")),
        **{column: item.get(column) for column in df.columns if column in item}
    } for index, item in results.items()]).reindex(columns=df.columns)])
    df.to_csv(csv_path, index=False, encoding="utf-8")


def pick_relevant(system_message, chunk):
    try:
        user_message = json.dumps(chunk[["title", "summary"]].to_dict("index"), ensure_ascii=False)
        indices = internal_text_chat("gpt4o_mini_openrouter", system_message, user_message)
        indices = eval(indices[indices.find("["):indices.find("]") + 1]) if "[" in indices and "]" in indices else []
        return chunk.loc[chunk.index.isin(indices)].to_dict("index")
    except Exception as e:
        print(f"Error in pick_relevant: {e}")
        return {}


def screen_search_results(csv_path, query):
    try:
        system_message = get_prompt("are_relevant_search_results", query=query)
        df = pd.read_csv(csv_path, encoding="utf-8")
        df["sequence"] = range(1, len(df) + 1)
        df = df.set_index("sequence")
        chunks = [df.iloc[i:i + 30] for i in range(0, len(df), 30)]
        requests = [(pick_relevant, system_message, chunk) for chunk in chunks]
        results = {key: value for result, name, arguments in manage_thread(requests) if result for key, value in result.items()}
        if results:
            csv_for_relevant_search_results(csv_path)
            relevant_search_results_to_csv(csv_path, results)
            return csv_path
        return None
    except Exception as e:
        print(f"Error in screen_search_results: {e}")
        return None


def pick_applicable(system_message, response_format, record, credible_only=False):
    try:
        if (web_text := record.get("web_text")) and (url := record.get("url")):
            user_message = json.dumps({"web_text": web_text, "url": url}, ensure_ascii=False)
            knowledge_extraction = internal_text_chat("gpt4o_openrouter", system_message, user_message, response_format)
            knowledge_extraction = json.loads(knowledge_extraction)
            if knowledge_extraction["is_applicable"] and (not credible_only or knowledge_extraction["is_credible"]):
                return {"applicable_knowledge": knowledge_extraction["applicable_knowledge"], **record}
            return {}
    except Exception as e:
        print(f"Error in pick_applicable: {e}")
        return {}


def filter_search_results(csv_path, query, credible_only=False):
    try:
        system_message = get_prompt("is_applicable_search_result", query=query)
        response_format = get_response_format("is_applicable_search_result_json")
        records = pd.read_csv(csv_path, encoding="utf-8").to_dict("records")
        requests = [(pick_applicable, system_message, response_format, record, credible_only) for record in records]
        results = {index: result for index, (result, name, arguments) in enumerate(manage_thread(requests)) if result}
        if results:
            csv_for_applicable_search_results(csv_path)
            applicable_search_results_to_csv(csv_path, results)
            return csv_path
        return None
    except Exception as e:
        print(f"Error in filter_search_results: {e}")
        return None


def get_applicable_knowledge(csv_path):
    try:
        return json.dumps({
            **{f"{index + 1}": {"applicable_knowledge": row.get("applicable_knowledge"), "title": row.get("title"), "url": row.get("url")}
               for index, row in enumerate(pd.read_csv(csv_path, encoding="utf-8").to_dict("records"))},
            "url_for_downloading_all_results": upload_to_container(csv_path)
        }, ensure_ascii=False)
    except Exception as e:
        print(f"Error in get_applicable_knowledge: {e}")
        return None


def basic_search(query):
    results = google_answer(query)
    if results:
        return results
    return None


def broader_search(query):
    requests = [
        (tavily_answer, query),
        (reader_search, query)
    ]
    results = "\n".join([result for result, name, arguments in manage_thread(requests) if result])
    if results:
        return results
    return None


def news_search_by_freshness(query, freshness=None, credible_only=False):
    freshness = freshness if isinstance(freshness, int) else None
    credible_only = credible_only if isinstance(credible_only, bool) else False
    csv_path = csv_for_search_results(query)
    requests = [
        (tavily_news_by_freshness, query, freshness),
        (exa_news_by_freshness, query, freshness)
    ]
    results = {key: value for result, name, arguments in manage_thread(requests) if result for key, value in result.items()}
    if results:
        search_results_to_csv(csv_path, results, freshness)
        if screen_search_results(csv_path, query):
            if filter_search_results(csv_path, query, credible_only):
                applicable_knowledge = get_applicable_knowledge(csv_path)
                if applicable_knowledge:
                    return applicable_knowledge
    return None


def scholar_search_by_freshness(query, freshness=None):
    freshness = freshness if isinstance(freshness, int) else None
    csv_path = csv_for_search_results(query)
    requests = [
        (exa_paper_by_freshness, query, freshness)
    ]
    results = {key: value for result, name, arguments in manage_thread(requests) if result for key, value in result.items()}
    if results:
        search_results_to_csv(csv_path, results, freshness)
        if screen_search_results(csv_path, query):
            if filter_search_results(csv_path, query):
                applicable_knowledge = get_applicable_knowledge(csv_path)
                if applicable_knowledge:
                    return applicable_knowledge
    return None


def serious_search_by_freshness(query, freshness=None, credible_only=False):
    freshness = freshness if isinstance(freshness, int) else None
    credible_only = credible_only if isinstance(credible_only, bool) else False
    csv_path = csv_for_search_results(query)
    requests = [
        (bing_by_freshness, query, freshness),
        (duckduckgo_by_freshness, query, freshness),
        (google_by_freshness, query, freshness),
        (exa_by_freshness, query, freshness)
    ]
    results = {key: value for result, name, arguments in manage_thread(requests) if result for key, value in result.items()}
    if results:
        search_results_to_csv(csv_path, results, freshness)
        if screen_search_results(csv_path, query):
            if filter_search_results(csv_path, query, credible_only):
                applicable_knowledge = get_applicable_knowledge(csv_path)
                if applicable_knowledge:
                    return applicable_knowledge
    return None


def scholar_search_by_year_range(query, year_range=None):
    year_range = year_range if isinstance(year_range[0], int) and isinstance(year_range[1], int) else None
    csv_path = csv_for_search_results(query)
    requests = [
        (google_scholar_by_year_range, query, year_range),
        (exa_paper_by_year_range, query, year_range)
    ]
    results = {key: value for result, name, arguments in manage_thread(requests) if result for key, value in result.items()}
    if results:
        search_results_to_csv(csv_path, results)
        if screen_search_results(csv_path, query):
            if filter_search_results(csv_path, query):
                applicable_knowledge = get_applicable_knowledge(csv_path)
                if applicable_knowledge:
                    return applicable_knowledge
    return None


def patents_search_by_year_range(query, year_range=None):
    year_range = year_range if isinstance(year_range[0], int) and isinstance(year_range[1], int) else None
    csv_path = csv_for_search_results(query)
    requests = [
        (google_patents_by_year_range, query, year_range)
    ]
    results = {key: value for result, name, arguments in manage_thread(requests) if result for key, value in result.items()}
    if results:
        search_results_to_csv(csv_path, results)
        if screen_search_results(csv_path, query):
            if filter_search_results(csv_path, query):
                applicable_knowledge = get_applicable_knowledge(csv_path)
                if applicable_knowledge:
                    return applicable_knowledge
    return None


def serious_search_by_year_range(query, year_range=None, credible_only=False):
    year_range = year_range if isinstance(year_range[0], int) and isinstance(year_range[1], int) else None
    credible_only = credible_only if isinstance(credible_only, bool) else False
    csv_path = csv_for_search_results(query)
    requests = [
        (bing_by_year_range, query, year_range),
        (duckduckgo_by_year_range, query, year_range),
        (exa_by_year_range, query, year_range)
    ]
    results = {key: value for result, name, arguments in manage_thread(requests) if result for key, value in result.items()}
    if results:
        search_results_to_csv(csv_path, results)
        if screen_search_results(csv_path, query):
            if filter_search_results(csv_path, query, credible_only):
                applicable_knowledge = get_applicable_knowledge(csv_path)
                if applicable_knowledge:
                    return applicable_knowledge
    return None


def resize_image(image_path, side_limit):
    with Image.open(image_path) as f:
        longer_side = max(f.size)
        if longer_side > side_limit:
            width = int(f.size[0] * side_limit / longer_side)
            height = int(f.size[1] * side_limit / longer_side)
            if f.size != (width, height):
                f = f.resize((width, height), Image.LANCZOS)
        f.save(image_path, f.format)  # 保存图片，保持原始格式


def is_plain_text(file_path, side_limit=1024):
    image_paths = []
    with fitz.open(file_path) as pdf:
        for i in range(pdf.page_count):
            image_path = f"temp-images/{now_in_filename()}.png"
            pdf.load_page(i).get_pixmap().save(image_path)
            if os.path.exists(image_path):
                image_paths.append(image_path)
            else:
                image_paths.append(None)

    if image_paths:
        requests = []
        for image_path in image_paths:
            if image_path:  # 仅处理有效的 image_path
                request = (resize_image, image_path, side_limit)  # 创建请求，传递图片路径和最大尺寸参数
                requests.append(request)
        manage_thread(requests)

        ais = ["qwen2_vl_7b_siliconflow", "qwen2_vl_7b_openrouter", "qwen_vl_plus_dashscope"]
        user_message = get_prompt("is_plain_text")
        requests = []
        for i, image_path in enumerate(image_paths, start=1):
            if image_path:
                request = (internal_image_chat, ais[(i - 1) % len(ais)], user_message, image_path)
                requests.append(request)
        results = {arguments[-1]: result for result, name, arguments in manage_thread(requests)}
        return [image_path if image_path and results.get(image_path) == "False" else None for image_path in image_paths]
    return None


def describe_visual_elements(file_path, ais):
    image_paths = is_plain_text(file_path)
    if image_paths:
        user_message = get_prompt("describe_visual_elements")
        requests = []
        for i, image_path in enumerate(image_paths, start=1):
            if image_path:
                request = (internal_image_chat, ais[(i - 1) % len(ais)], user_message, image_path)
                requests.append(request)
        results = {arguments[-1]: result for result, name, arguments in manage_thread(requests)}
        return [results.get(image_path) if image_path else None for image_path in image_paths]
    return None


def extract_page_texts(file_path, pages_per_chunk, overlap_length=100):
    for attempt, config in enumerate(random.sample(DOCUMENT_CONFIGS, len(DOCUMENT_CONFIGS)), 1):
        try:
            client = DocumentAnalysisClient(endpoint=config["endpoint"], credential=AzureKeyCredential(config["api_key"]))
            with open(file_path, "rb") as f:
                page_texts = ["\n".join([line.content for line in page.lines]) for page in client.begin_analyze_document("prebuilt-read", f).result().pages]
                for i in range(len(page_texts)):
                    if i > 0 and i % pages_per_chunk == 0:
                        prev_text = page_texts[i - 1]
                        overlap_text = prev_text[-overlap_length:] if len(prev_text) > overlap_length else prev_text
                        page_texts[i] = overlap_text + " " + page_texts[i]
                return page_texts
        except Exception as e:
            print(f"Attempt {attempt} failed: {e}")

    print("Falling back to local processing")
    try:
        with fitz.open(file_path) as f:
            page_texts = [page.get_text() for page in f]
            for i in range(len(page_texts)):
                if i > 0 and i % pages_per_chunk == 0:
                    prev_text = page_texts[i - 1]
                    overlap_text = prev_text[-overlap_length:] if len(prev_text) > overlap_length else prev_text
                    page_texts[i] = overlap_text + " " + page_texts[i]
            return page_texts
    except Exception as e:
        print(f"Local processing failed: {e}")
        return None


def parse_pdf_files(files, pages_per_chunk, overlap_length, is_plain_text=False, ais=None):
    contents = {}
    for file in files:
        file_path = f"uploaded-files/{sanitize_filename(os.path.splitext(file.name)[0])} {now_in_filename()}{os.path.splitext(file.name)[1]}"
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        if is_plain_text:
            page_texts = extract_page_texts(file_path, pages_per_chunk, overlap_length)
            if page_texts:
                content = {i // pages_per_chunk + 1: "\n\n".join(page_text for page_text in page_texts[i:i + pages_per_chunk]) for i in range(0, len(page_texts), pages_per_chunk)}
                contents.update({(len(contents) + key): value for key, value in content.items()})
        elif ais:
            requests = [
                (describe_visual_elements, file_path, ais),
                (extract_page_texts, file_path, pages_per_chunk, overlap_length)
            ]
            results = {name: result for result, name, arguments in manage_thread(requests)}
            page_descriptions = results.get("describe_visual_elements")
            page_texts = results.get("extract_page_texts")
            if page_descriptions and page_texts:
                content = {i // pages_per_chunk + 1: "\n\n".join((page_descriptions[i + j] + "\n\n" if i + j < len(page_descriptions) and page_descriptions[i + j] else "") + page_text for j, page_text in enumerate(page_texts[i:i + pages_per_chunk])) for i in range(0, len(page_texts), pages_per_chunk)}
                contents.update({(len(contents) + key): value for key, value in content.items()})
    return contents


def parse_txt_files(files, chars_per_chunk, overlap_length):
    contents = {}
    for file in files:
        file_path = f"uploaded-files/{sanitize_filename(os.path.splitext(file.name)[0])} {now_in_filename()}{os.path.splitext(file.name)[1]}"
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            chunk_texts = [content[i:i + chars_per_chunk] for i in range(0, len(content), chars_per_chunk)]
            for i in range(1, len(chunk_texts)):
                prev_text = chunk_texts[i - 1]
                overlap_text = prev_text[-overlap_length:] if len(prev_text) > overlap_length else prev_text
                chunk_texts[i] = overlap_text + " " + chunk_texts[i]
            content = {i + 1: chunk_text for i, chunk_text in enumerate(chunk_texts)}
            contents.update({(len(contents) + key): value for key, value in content.items()})
    return contents


def get_doc_contents(txt_path):
    with open(txt_path, "r", encoding="utf-8") as f:
        return eval(f.read())


def extract_result(text, tag):
    matched = re.search(f"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    if matched:
        return matched.group(1).strip()
    else:
        return re.sub(r'<[^>]+>', '', text).strip()


def parse_result(text, pattern):
    markers = [(marker.start(), marker.group()) for marker in re.finditer(pattern, text)]
    if markers:
        return {i + 1: text[marker[0] + len(marker[1]):markers[i + 1][0] if i < len(markers) - 1 else len(text)].strip() for i, marker in enumerate(markers)}
    else:
        return None


def filter_words(text, words):
    for word in words:
        text = text.replace(word, '')
    return text


def azure_tts(text, voice_gender):
    wav_path = f"temp-data/{now_in_filename()}.wav"
    for attempt, config in enumerate(random.sample(SPEECH_CONFIGS, len(SPEECH_CONFIGS)), 1):
        try:
            speech_config = speechsdk.SpeechConfig(config["api_key"], config["region"])
            speech_config.speech_synthesis_voice_name = "zh-CN-YunxiaoMultilingualNeural" if voice_gender == "male" else "zh-CN-XiaoyuMultilingualNeural"
            synthesizer = speechsdk.SpeechSynthesizer(speech_config, speechsdk.AudioConfig(filename=wav_path))
            if synthesizer.speak_text_async(text).get():
                return wav_path
        except Exception as e:
            print(f"Attempt {attempt} failed: {e}")
    print("Failed to synthesize speech after maximum retries")
    return None


def generate_audios_serially(texts, voice_gender):
    return {key: azure_tts(text, voice_gender) for key, text in texts.items()}


def merge_audios(audios):
    wav_path = f"temp-data/{now_in_filename()}.wav"
    audios = [audios[key] for key in sorted(audios) if audios[key] is not None]
    with wave.open(audios[0], "rb") as wav:
        params = wav.getparams()
    with wave.open(wav_path, "wb") as output:
        output.setparams(params)
        for audio in audios:
            with wave.open(audio, "rb") as wav:
                output.writeframes(wav.readframes(wav.getnframes()))
    return wav_path


def outline_for_audio_interpretation(csv_path, user_requirements, doc_content):
    system_message = get_prompt("outline_for_audio_interpretation")
    user_message = f"<user_requirements>\n{user_requirements}\n</user_requirements>\n<doc_content>\n{doc_content}\n</doc_content>"
    outline = extract_result(internal_text_chat("gpt4o_mini_excellence2", system_message, user_message), "outline")
    headings_and_notes = parse_result(outline, r"Dialogue Question of Chapter \d+:|Notes:")
    if isinstance(headings_and_notes, dict):
        *headings, notes = headings_and_notes.items()
        df = pd.DataFrame(columns=list(dict(headings).values()))
        if os.path.exists(csv_path):
            df = pd.concat([pd.read_csv(csv_path), df], axis=1)
        df.to_csv(csv_path, index=False, encoding="utf-8")
        return outline, None if notes[1] == "None" else notes[1]
    return None


def extract_relevant_information(key, doc_content, outline, ai, system_message):
    user_message = f"<outline>\n{outline}\n</outline>\n<doc_content>\n{doc_content}\n</doc_content>"
    return parse_result(extract_result(internal_text_chat(ai, system_message, user_message), "information_for_questions"), r"Information for Question \d+:")


def information_for_audio_interpretation(doc_contents, outline):
    ais = ["gpt4o_mini_openrouter", "gpt4o_mini_rainboweye", "gpt4o_mini_excellence2"]
    system_message = get_prompt("information_for_audio_interpretation")
    requests = []
    for key, doc_content in doc_contents.items():
        request = (extract_relevant_information, key, doc_content, outline, ais[(key - 1) % len(ais)], system_message)
        requests.append(request)
    results = {arguments[0]: result for result, name, arguments in manage_thread(requests)}
    return results


def information_for_questions_to_csv(csv_path, information_for_questions):
    df = pd.read_csv(csv_path)
    start_column = next((i for i, column in enumerate(df.columns) if df[column].isna().all()), 0)
    df = pd.concat([df, pd.DataFrame([
        [None] * start_column + [None if information_for_question[key] == "None" else information_for_question[key]
         for key in range(1, len(information_for_question) + 1)]
        for key, information_for_question in sorted(information_for_questions.items())], columns=df.columns)])
    df.to_csv(csv_path, index=False, encoding="utf-8")


def craft_interpretation_script(i, column, information_for_question, notes, ai, system_message):
    default_notes = """Pay attention to pivotal findings and their business relevance. Decode complex ideas using everyday examples and comparisons. Provide contextual explanations for technical terms. Conclude with key takeaways and a final statement."""
    user_message = f"<question>\n{column}\n</question>\n<information_for_question>\n{information_for_question}\n</information_for_question>\n<notes>\n{notes if notes else default_notes}\n</notes>"
    return filter_words(extract_result(internal_text_chat(ai, system_message, user_message), "script"), ["首先，", "其次，", "再次，", "最后，", "然而，", "然而", "此外，", "此外", "除此之外，", "总之，", "总而言之，", "综上所述，", "中共", "中国共产党"])


def scripts_for_audio_interpretation(csv_path, notes):
    ais = ["gpt4o_mini_openrouter", "gpt4o_mini_rainboweye", "gpt4o_mini_excellence2"]
    system_message = get_prompt("script_for_audio_interpretation")
    df = pd.read_csv(csv_path)
    requests = []
    for i, column in enumerate(df.columns, start=1):
        information_for_question = "\n\n".join(df[column].dropna().astype(str))
        if len(information_for_question) >= 300:
            request = (craft_interpretation_script, i, column, information_for_question, notes, ais[(i - 1) % len(ais)], system_message)
            requests.append(request)
    results = {arguments[0]: f"{arguments[1]}\n{result}" for result, name, arguments in manage_thread(requests)}
    return results


def generate_audio_interpretation(txt_path, user_requirements, voice_gender, to_email=None):
    csv_path = f"temp-data/{now_in_filename()}.csv"
    doc_contents = get_doc_contents(txt_path)
    all_content = "\n\n".join(doc_contents[key] for key in sorted(doc_contents))[:100000]
    for attempt in range(3):
        try:
            outline, notes = outline_for_audio_interpretation(csv_path, user_requirements, all_content)
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
                                                if audios and sum(1 for audio in audios.values() if audio is None) / len(audios) < 0.2:
                                                    for attempt in range(3):
                                                        try:
                                                            wav_path = merge_audios(audios)
                                                            if wav_path:
                                                                file_url = upload_and_send(wav_path, to_email)
                                                                return file_url
                                                        except Exception:
                                                            print(f"Attempt {attempt + 1} failed")
                                                    print("Failed at step 5 after maximum retries")
                                                    return None
                                            except Exception:
                                                print(f"Attempt {attempt + 1} failed")
                                        print("Failed at step 4 after maximum retries")
                                        return None
                                except Exception:
                                    print(f"Attempt {attempt + 1} failed")
                            print("Failed at step 3 after maximum retries")
                            return None
                    except Exception:
                        print(f"Attempt {attempt + 1} failed")
                print("Failed at step 2 after maximum retries")
                return None
        except Exception:
            print(f"Attempt {attempt + 1} failed")
    print("Failed at step 1 after maximum retries")
    return None


def outline_for_audio_storytelling(csv_path, user_requirements, doc_content):
    system_message = get_prompt("outline_for_audio_storytelling")
    user_message = f"<user_requirements>\n{user_requirements}\n</user_requirements>\n<doc_content>\n{doc_content}\n</doc_content>"
    outline = extract_result(internal_text_chat("gpt4o_mini_excellence2", system_message, user_message), "outline")
    headings_and_notes = parse_result(outline, r"Heading and Plot Points of Chapter \d+:|Notes:")
    if isinstance(headings_and_notes, dict):
        *headings, notes = headings_and_notes.items()
        df = pd.DataFrame(columns=list(dict(headings).values()))
        if os.path.exists(csv_path):
            df = pd.concat([pd.read_csv(csv_path), df], axis=1)
        df.to_csv(csv_path, index=False, encoding="utf-8")
        return None if notes[1] == "None" else notes[1]
    return None


def craft_storytelling_script(column, completed_scripts, doc_content, notes, ai, system_message):
    default_notes = """Pay attention to pivotal scenes. Guide the audience through the tale, incorporating rhetorical techniques and cultural insights when appropriate. Harmonise plot and character development through targeted details."""
    user_message = f"<heading_and_plot_points>\n{column}\n</heading_and_plot_points>\n<completed_scripts>\n{completed_scripts}\n</completed_scripts>\n<doc_content>\n{doc_content}\n</doc_content>\n<notes>\n{notes if notes else default_notes}\n</notes>"
    return filter_words(extract_result(internal_text_chat(ai, system_message, user_message), "script"), ["首先，", "其次，", "再次，", "最后，", "然而，", "然而", "此外，", "此外", "除此之外，", "总之，", "总而言之，", "综上所述，", "中共", "中国共产党"])


def scripts_for_audio_storytelling(csv_path, doc_content, notes):
    ais = ["gpt4o_mini_openrouter", "gpt4o_mini_rainboweye", "gpt4o_mini_excellence2"]
    system_message = get_prompt("script_for_audio_storytelling")
    df = pd.read_csv(csv_path)
    if len(df) == 0:
        df.loc[0] = pd.NA
    for i, column in enumerate(df.columns, start=1):
        if pd.isna(df.iloc[0, i - 1]):
            completed_scripts = "\n\n".join(df.iloc[0, :i - 1].values.astype(str)) if i > 1 else ""
            script = craft_storytelling_script(column, completed_scripts, doc_content, notes, ais[(i - 1) % len(ais)], system_message)
            if script:
                df.iloc[0, i - 1] = script
                df.to_csv(csv_path, index=False, encoding="utf-8")
    return {i: script for i, script in enumerate(df.iloc[0].values, start=1)}


def generate_audio_storytelling(txt_path, user_requirements, voice_gender, to_email=None):
    csv_path = f"temp-data/{now_in_filename()}.csv"
    doc_contents = get_doc_contents(txt_path)
    for doc_content in doc_contents.values():
        for attempt in range(3):
            try:
                notes = outline_for_audio_storytelling(csv_path, user_requirements, doc_content)
                if csv_path:
                    for attempt in range(3):
                        try:
                            scripts = scripts_for_audio_storytelling(csv_path, doc_content, notes)
                            if scripts:
                                for attempt in range(3):
                                    try:
                                        audios = generate_audios_serially(scripts, voice_gender)
                                        if audios and sum(1 for audio in audios.values() if audio is None) / len(audios) < 0.2:
                                            for attempt in range(3):
                                                try:
                                                    wav_path = merge_audios(audios)
                                                    if wav_path:
                                                        file_url = upload_and_send(wav_path, to_email)
                                                        return file_url
                                                except Exception:
                                                    print(f"Attempt {attempt + 1} failed")
                                            print("Failed at step 4 after maximum retries")
                                            return None
                                    except Exception:
                                        print(f"Attempt {attempt + 1} failed")
                                print("Failed at step 3 after maximum retries")
                                return None
                        except Exception:
                            print(f"Attempt {attempt + 1} failed")
                    print("Failed at step 2 after maximum retries")
                    return None
            except Exception:
                print(f"Attempt {attempt + 1} failed")
        print("Failed at step 1 after maximum retries")
        return None


def calculator(expression):
    getcontext().prec = 50
    try:
        return eval(expression, {"__builtins__": None, "math": math, "Decimal": Decimal})
    except Exception:
        return None
