from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
import azure.cognitiveservices.speech as speechsdk
import os
import requests
import importlib
from itertools import permutations
import base64
import pandas as pd
import json
import random
import fitz
import wave
import math
from decimal import Decimal, getcontext
from aife_utils import retrieve, manage_futures, manage_futures_async, upload_to_container, resize_images, pdf_to_images, save_as_txt, get_text, extract_text_with_xml_tags, extract_text_with_pattern, filter_words, now_and_choices


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
MAIDSR1_API_KEY = retrieve("MAIDSR1Key")
R1_API_KEY = retrieve("DeepSeekR1Key")
OCR_API_KEY = retrieve("MistralOCRKey")

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
maidsr1 = LLM("https://MAI-DS-R1-yusi.westus3.models.ai.azure.com/chat/completions", MAIDSR1_API_KEY)
r1 = LLM("https://DeepSeek-R1-yusi.westus3.models.ai.azure.com/chat/completions", R1_API_KEY)
ocr = LLM("https://mistral-ocr-yusi.westus3.models.ai.azure.com/chat/completions", OCR_API_KEY)


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
    return None


def text_chats(configs, system_message, user_messages):
    configs = {i: list(permutation) for i, permutation in enumerate(permutations(configs))}
    requests = [(text_chat, (configs[i % len(configs)], system_message, user_message, i)) for i, user_message in enumerate(user_messages if isinstance(user_messages, list) else [user_messages], start=0)]
    return {arguments[-1]: result for result, function, arguments in manage_futures(requests)}


def image_chat(configs, user_message, image_path):
    if image_path:
        messages = [{"role": "user", "content": [{"type": "text", "text": user_message}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64.b64encode(open(image_path, 'rb').read()).decode('utf-8')}"}}]}]
        return route_llm_request(configs, messages)
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
        "llms": ["DeepSeek-V3-0324", "Qwen-Max", "Microsoft: MAI DS R1", "DeepSeek-R1", "QwQ 32B", "MiniMax-01", "Grok 3", "Gemini Pro 2.5", "Gemini Flash 2.5", "Gemma 3 27B", "Mistral Small 3.1", "Phi 4", "Claude 3.7 Sonnet", "o4 mini high", "GPT-4.1", "Yi-Lightning", "LFM 7B"],
        "system_message": "chat_only",
        "tools": None,
        "intro": "体验全球第一梯队的大语言模型"
    },
    "Text Chat with Search": {
        "llms": ["DeepSeek-R1 Search", "DeepSeek-R1 Research"],
        "system_message": "perplexity_chat_and_search",
        "tools": None,
        "intro": "支持搜索和深度研究"
    },
    "Multimodal Chat with Tools": {
        "llms": ["MiniMax-01", "Gemini Pro 2.5", "Gemini Flash 2.5", "Claude 3.7 Sonnet", "GPT-4.1"],
        "system_message": "chat_with_tools",
        "tools": ["generate_column_article_func", "generate_audio_interpretation_func", "calculator_func"],
        "intro": "生成专栏文章 生成音频解读 使用计算器"
    }
}

llms = {
    "QwQ 32B": {
        "category": "text",
        "configs": ["qwq_32b_deepinfra", "qwq_32b_hyperbolic", "qwq_32b_openrouter"],
        "context_length": 131072
    },
    "Qwen-Max": {
        "category": "text",
        "configs": ["qwen_max_dashscope"],
        "context_length": 32768
    },
    "Microsoft: MAI DS R1": {
        "category": "text",
        "configs": ["mai_ds_r1_openrouter", "mai_ds_r1_azure"],
        "context_length": 163840
    },
    "DeepSeek-R1": {
        "category": "text",
        "configs": ["deepseek_r1_hyperbolic", "deepseek_r1_dashscope", "deepseek_r1_azure", "deepseek_r1", "deepseek_r1_luchen", "deepseek_r1_luchen2"],
        "context_length": 65536
    },
    "DeepSeek-V3-0324": {
        "category": "text",
        "configs": ["deepseek_v3_openrouter", "deepseek_v3_hyperbolic", "deepseek_v3_deepinfra", "deepseek_v3"],
        "context_length": 65536
    },
    "DeepSeek-R1 Search": {
        "category": "text",
        "configs": ["sonar_reasoning_pro_openrouter"],
        "context_length": 128000
    },
    "DeepSeek-R1 Research": {
        "category": "text",
        "configs": ["sonar_deep_research_openrouter"],
        "context_length": 128000
    },
    "MiniMax-01": {
        "category": "multimodal",
        "configs": ["minimax_01", "minimax_01_openrouter"],
        "context_length": 1000192
    },
    "Gemini Pro 2.5": {
        "category": "multimodal",
        "configs": ["gemini_25_pro_openrouter"],
        "context_length": 1048576
    },
    "Gemini Flash 2.5": {
        "category": "multimodal",
        "configs": ["gemini25_flash_openrouter"],
        "context_length": 1048576
    },
    "Gemma 3 27B": {
        "category": "multimodal",
        "configs": ["gemma3_27b_openrouter"],
        "context_length": 131072
    },
    "Mistral Small 3.1": {
        "category": "text",
        "configs": ["mistral_small_31_openrouter"],
        "context_length": 128000
    },
    "Phi 4": {
        "category": "text",
        "configs": ["phi4_deepinfra", "phi4_openrouter"],
        "context_length": 16384
    },
    "Grok 3": {
        "category": "text",
        "configs": ["grok3_xai", "grok3_openrouter"],
        "context_length": 131072
    },
    "o4 mini high": {
        "category": "text",
        "configs": ["o4_mini_high_openrouter"],
        "context_length": 200000
    },
    "GPT-4.1": {
        "category": "multimodal",
        "configs": ["gpt41_openrouter"],
        "context_length": 1047576
    },
    "Claude 3.7 Sonnet": {
        "category": "multimodal",
        "configs": ["claude37_sonnet_openrouter"],
        "context_length": 200000
    },
    "Yi-Lightning": {
        "category": "text",
        "configs": ["yi_lightning"],
        "context_length": 16384
    },
    "LFM 7B": {
        "category": "text",
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
    "mai_ds_r1_openrouter": {
        "name": "openrouter",
        "arguments": {
            "model": "microsoft/mai-ds-r1:free",
            "temperature": 0.15,
            "top_p": 0.95,
            "max_tokens": 8192
        }
    },
    "mai_ds_r1_azure": {
        "name": "maidsr1",
        "arguments": {
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
    "grok3_xai": {
        "name": "xai",
        "arguments": {
            "model": "grok-3-beta",
            "temperature": 0.15,
            "top_p": 0.95,
            "max_tokens": 8192
        }
    },
    "grok3_openrouter": {
        "name": "openrouter",
        "arguments": {
            "model": "x-ai/grok-3-beta",
            "temperature": 0.15,
            "top_p": 0.95,
            "max_tokens": 8192
        }
    },
    "o4_mini_high_openrouter": {
        "name": "openrouter",
        "arguments": {
            "model": "openai/o4-mini-high",
            "temperature": 0.15,
            "top_p": 0.95,
            "max_tokens": 8192
        }
    },
    "gpt41_openrouter": {
        "name": "openrouter",
        "arguments": {
            "model": "openai/gpt-4.1",
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
            "model": "google/gemini-2.5-pro-preview-03-25",
            "temperature": 0.15,
            "top_p": 0.95,
            "max_tokens": 8192
        }
    },
    "gemini25_flash_openrouter": {
        "name": "openrouter",
        "arguments": {
            "model": "google/gemini-2.5-flash-preview:thinking",
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

generate_column_article_func = {
    "type": "function",
    "function": {
        "name": "generate_column_article",
        "description": "Generate a column article in accordance with the user's requirements and provided documents. This function returns a URL for downloading the text file of the article.",
        "parameters": {
            "type": "object",
            "properties": {
                "user_requirements": {
                    "type": "string",
                    "description": "The user's requirements for the article, such as concerns or opinions."
                },
                "txt_paths": {
                    "type": "array",
                    "description": "A list of paths to TXT files, each containing a doc_content. These paths can be found in the chat history.",
                    "items": {
                        "type": "string",
                        "description": "The path to the TXT file containing the doc_content."
                    },
                }
            },
            "required": ["user_requirements", "txt_paths"],
            "additionalProperties": False,
        },
    },
}

generate_audio_interpretation_func = {
    "type": "function",
    "function": {
        "name": "generate_audio_interpretation",
        "description": "Generate an audio interpretation in accordance with the user's requirements and provided documents. This function returns a URL for downloading the audio file of the interpretation.",
        "parameters": {
            "type": "object",
            "properties": {
                "user_requirements": {
                    "type": "string",
                    "description": "The user's requirements for the interpretation, such as concerns or interests."
                },
                "txt_paths": {
                    "type": "array",
                    "description": "A list of paths to TXT files, each containing a doc_content. These paths can be found in the chat history.",
                    "items": {
                        "type": "string",
                        "description": "The path to the TXT file containing the doc_content."
                    },
                },
                "voice_gender": {
                    "type": "string",
                    "enum": ["male", "female"],
                    "description": "The user's choice of male or female voice for the audio."
                },
            },
            "required": ["user_requirements", "txt_paths", "voice_gender"],
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


def describe_pages(pdf_path):
    if image_paths := pdf_to_images(pdf_path):
        if any(image_paths):
            resize_images(image_paths, 1280, False)
            llm_results = image_chats(["qwen25_vl_7b_hyperbolic", "qwen25_vl_7b_dashscope", "qwen_vl_plus_dashscope"], get_prompt("is_plain_text"), image_paths)
            image_paths = [image_path if image_path and llm_results[image_path] and ("False" in llm_results[image_path] or "false" in llm_results[image_path]) else None for image_path in image_paths]
            if any(image_paths):
                llm_results = image_chats(["qwen25_vl_72b_hyperbolic", "qwen25_vl_72b_dashscope", "qwen_vl_max_dashscope"], get_prompt("describe_visual_elements"), image_paths)
                return [llm_results[image_path] if image_path and llm_results[image_path] and len(llm_results[image_path]) > 50 else None for image_path in image_paths]


def extract_page_texts(file_path, pages_per_chunk, overlap_length):
    for attempt, config in enumerate(random.sample(DOCUMENT_CONFIGS, len(DOCUMENT_CONFIGS)), 1):
        try:
            client = DocumentAnalysisClient(endpoint=config["endpoint"], credential=AzureKeyCredential(config["api_key"]))
            with open(file_path, "rb") as f:
                page_texts = ["\n".join([line.content for line in page.lines]) for page in client.begin_analyze_document("prebuilt-read", f).result().pages]
                for i in range(len(page_texts)):
                    if i > 0 and i % pages_per_chunk == 0:
                        prev_text = page_texts[i - 1]
                        overlap_text = prev_text[-overlap_length:] if len(prev_text) > overlap_length else ""
                        page_texts[i] = overlap_text + " " + page_texts[i]
                return page_texts
        except Exception:
            continue
    print("Falling back to local processing")
    try:
        with fitz.open(file_path) as f:
            page_texts = [page.get_text() for page in f]
            for i in range(len(page_texts)):
                if i > 0 and i % pages_per_chunk == 0:
                    prev_text = page_texts[i - 1]
                    overlap_text = prev_text[-overlap_length:] if len(prev_text) > overlap_length else ""
                    page_texts[i] = overlap_text + " " + page_texts[i]
            return page_texts
    except Exception as e:
        print(f"Local processing failed: {e}")
        return None


def parse_pdfs(pdf_paths, pdf_type, to_txt=False):
    contents = []
    if pdf_type == "Plain Text":
        pages_per_chunk = 4
        overlap_length = 1000
        for pdf_path in pdf_paths:
            if page_texts := extract_page_texts(pdf_path, pages_per_chunk, overlap_length):
                page_contents = ["\n\n".join(page_text for page_text in page_texts[i:i + pages_per_chunk]) for i in range(0, len(page_texts), pages_per_chunk)]
                contents.extend(page_contents)
    elif pdf_type == "With Visuals" or pdf_type is None:
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
                    page_contents = ["\n\n".join((page_descriptions[i + j] + "\n\n" if i + j < len(page_descriptions) and page_descriptions[i + j] else "") + page_text for j, page_text in enumerate(page_texts[i:i + pages_per_chunk])) for i in range(0, len(page_texts), pages_per_chunk)]
                else:
                    page_contents = ["\n\n".join(page_text for page_text in page_texts[i:i + pages_per_chunk]) for i in range(0, len(page_texts), pages_per_chunk)]
                contents.extend(page_contents)
    if contents:
        if to_txt:
            txt_path = save_as_txt(str(contents))
            return contents, txt_path
        return contents
    return None


def outline_for_column_article(csv_path, user_requirements, content):
    system_message = get_prompt("outline_for_column_article")
    user_message = f"<user_requirements>\n{user_requirements}\n</user_requirements>\n<doc_content>\n{content}\n</doc_content>"
    for attempt in range(3):
        try:
            llm_result = text_chat(["gemini_25_pro_openrouter"], system_message, user_message)
            outline = extract_text_with_xml_tags(llm_result, "outline")
            if subheadings_and_points := extract_text_with_pattern(outline, r"Subheading and Points of Chapter \d+:"):
                pd.DataFrame(columns=subheadings_and_points).to_csv(csv_path, index=False, encoding="utf-8")
                return outline
        except Exception:
            continue
    return None


def information_for_column_article(contents, outline):
    system_message = get_prompt("information_for_column_article")
    user_messages = [f"<outline>\n{outline}\n</outline>\n<doc_content>\n{content}\n</doc_content>" for content in contents]
    llm_results = text_chats(["deepseek_v3_openrouter", "deepseek_v3_hyperbolic", "deepseek_v3_deepinfra"], system_message, user_messages)
    return [extract_text_with_pattern(extract_text_with_xml_tags(llm_results[i], "information_for_subheadings"), r"Information for Subheading \d+:") for i in llm_results]


def information_for_subheadings_to_csv(csv_path, information_for_subheadings):
    df = pd.read_csv(csv_path, encoding="utf-8")
    df = pd.concat([df, pd.DataFrame([[None if information_in_pages == "None" else information_in_pages for information_in_pages in information_for_subheading[:len(df.columns)]] for information_for_subheading in information_for_subheadings], columns=df.columns)])
    df.to_csv(csv_path, index=False, encoding="utf-8")


def contents_for_column_article(csv_path):
    system_message = get_prompt("content_for_column_article")
    df = pd.read_csv(csv_path, encoding="utf-8")
    user_messages = [f"<subheading_and_points>\n{column}\n</subheading_and_points>\n<information_for_subheading>\n{information_for_subheading}\n</information_for_subheading>" for column in df.columns if len(information_for_subheading := "\n\n".join(df[column].dropna().astype(str))) >= 300]
    llm_results = text_chats(["deepseek_v3_openrouter", "deepseek_v3_hyperbolic", "deepseek_v3_deepinfra"], system_message, user_messages)
    return [f"{extract_text_with_xml_tags(llm_results[i], 'content')}" for i in llm_results]


def complete_column_article(outline, content):
    system_message = get_prompt("complete_column_article")
    user_message = f"<outline>\n{outline}\n</outline>\n<combined_content>\n{content}\n</combined_content>"
    for attempt in range(3):
        try:
            llm_result = text_chat(["deepseek_r1_azure", "deepseek_r1_hyperbolic", "deepseek_r1_dashscope", "deepseek_r1_luchen2", "deepseek_r1_luchen"], system_message, user_message)
            article_title, integrated_content, final_analysis_conclusion = extract_text_with_xml_tags(llm_result, ["article_title", "integrated_content", "final_analysis_conclusion"])
            column_article = f"{article_title}\n\n{integrated_content}\n\n{final_analysis_conclusion}"
            return filter_words(column_article, ["*", "首先，", "其次，", "再次，", "最后，", "然而，", "然而", "此外，", "此外", "除此之外，", "总之，", "总而言之", "总的来说", "综上所述", "中共", "中国共产党"])
        except Exception:
            continue
    return None


def generate_column_article(user_requirements, txt_paths):
    csv_path = os.path.join("temp-data", f"{now_and_choices()}.csv")
    contents = get_text(txt_paths)
    if contents:
        outline = outline_for_column_article(csv_path, user_requirements, "\n\n".join(contents))
        if outline:
            information_for_subheadings = information_for_column_article(contents, outline)
            if information_for_subheadings:
                information_for_subheadings_to_csv(csv_path, information_for_subheadings)
                contents = contents_for_column_article(csv_path)
                if contents:
                    column_article = complete_column_article(outline, "\n\n".join(contents))
                    if column_article:
                        txt_path = save_as_txt(column_article)
                        return upload_to_container(txt_path)
    return None


def outline_for_audio_interpretation(csv_path, user_requirements, content):
    system_message = get_prompt("outline_for_audio_interpretation")
    user_message = f"<user_requirements>\n{user_requirements}\n</user_requirements>\n<doc_content>\n{content}\n</doc_content>"
    for attempt in range(3):
        try:
            llm_result = text_chat(["gemini_25_pro_openrouter"], system_message, user_message)
            outline = extract_text_with_xml_tags(llm_result, "outline")
            if dialogue_questions := extract_text_with_pattern(outline, r"Dialogue Question of Chapter \d+:"):
                pd.DataFrame(columns=dialogue_questions).to_csv(csv_path, index=False, encoding="utf-8")
                return outline
        except Exception:
            continue
    return None


def information_for_audio_interpretation(contents, outline):
    system_message = get_prompt("information_for_audio_interpretation")
    user_messages = [f"<outline>\n{outline}\n</outline>\n<doc_content>\n{content}\n</doc_content>" for content in contents]
    llm_results = text_chats(["deepseek_v3_openrouter", "deepseek_v3_hyperbolic", "deepseek_v3_deepinfra"], system_message, user_messages)
    return [extract_text_with_pattern(extract_text_with_xml_tags(llm_results[i], "information_for_questions"), r"Information for Question \d+:") for i in llm_results]


def information_for_questions_to_csv(csv_path, information_for_questions):
    df = pd.read_csv(csv_path, encoding="utf-8")
    df = pd.concat([df, pd.DataFrame([[None if information_in_pages == "None" else information_in_pages for information_in_pages in information_for_question[:len(df.columns)]] for information_for_question in information_for_questions], columns=df.columns)])
    df.to_csv(csv_path, index=False, encoding="utf-8")


def scripts_for_audio_interpretation(csv_path):
    system_message = get_prompt("script_for_audio_interpretation")
    df = pd.read_csv(csv_path, encoding="utf-8")
    user_messages = [f"<dialogue_question>\n{column}\n</dialogue_question>\n<information_for_question>\n{information_for_question}\n</information_for_question>" for column in df.columns if len(information_for_question := "\n\n".join(df[column].dropna().astype(str))) >= 300]
    llm_results = text_chats(["deepseek_v3_openrouter", "deepseek_v3_hyperbolic", "deepseek_v3_deepinfra"], system_message, user_messages)
    return [f"{df.columns[i]}\n{filter_words(extract_text_with_xml_tags(llm_results[i], 'script'), ['*', '首先，', '其次，', '再次，', '最后，', '然而，', '然而', '此外，', '此外', '除此之外，', '总之，', '总而言之，', '总的来说，', '综上所述，', '中共', '中国共产党'])}" for i in llm_results]


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
    return [azure_tts(text, voice_gender) for text in texts]


def merge_audios(audio_paths):
    valid_audios = []
    for audio_path in audio_paths:
        try:
            with wave.open(audio_path, "rb") as f:
                if not valid_audios:
                    params = f.getparams()
                valid_audios.append(audio_path)
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


def generate_audio_interpretation(user_requirements, txt_paths, voice_gender):
    csv_path = os.path.join("temp-data", f"{now_and_choices()}.csv")
    contents = get_text(txt_paths)
    if contents:
        outline = outline_for_audio_interpretation(csv_path, user_requirements, "\n\n".join(contents))
        if outline:
            information_for_questions = information_for_audio_interpretation(contents, outline)
            if information_for_questions:
                information_for_questions_to_csv(csv_path, information_for_questions)
                scripts = scripts_for_audio_interpretation(csv_path)
                if scripts:
                    audios = generate_audios_serially(scripts, voice_gender)
                    if audios and sum(1 for audio in audios if not audio) / len(audios) < 0.2:
                        wav_path = merge_audios(audios)
                        if wav_path:
                            return upload_to_container(wav_path)
    return None


def calculator(expression):
    getcontext().prec = 50
    try:
        return eval(expression, {"__builtins__": None, "math": math, "Decimal": Decimal})
    except Exception:
        return None
