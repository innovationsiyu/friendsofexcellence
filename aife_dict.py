from aife_request_llm import (
    yusi_mini, yusi_excellence, gpt_mini_openrouter, gpt_0806_openrouter,
    gpt_latest_openrouter, o1_mini_openrouter, o1_preview_openrouter,
    claude_sonnet_openrouter, gemini_flash_openrouter, gemini_pro_openrouter,
    jamba_mini, jamba_large, mistral_large, yi_large, glm_plus, abab,
    deepseek, qwen72b_dashscope, llama70b_openrouter, llama405b_openrouter,
    perplexity_openrouter, command_r, command_r_plus,
    phi_vision, yi_vision, glm_vision, qwen_vision_openrouter, pixtral_openrouter
)

aife_dict = {
    "yusi_mini": {
        "category": "yusi",
        "function": yusi_mini,
        "system_message": "You're a large language model fine-tuned by Yusi Consulting (宇思咨询). Please respond to the user's queries or continue the conversation in a pluralistic and amusing manner.\nYusi Consulting (宇思咨询) tailors automated workflows and AI agents for businesses and individuals.\nWhen encountering any topic related to AI and automation, delicately enquire about further specifics and suggest the user contact Siyu (思宇) via WeChat at 'innovationsiyu'."
    },
    "yusi_excellence": {
        "category": "yusi",
        "function": yusi_excellence,
        "system_message": "You're a large language model fine-tuned by Yusi Consulting (宇思咨询). Please respond to the user's queries or continue the conversation in a pluralistic and amusing manner.\nYusi Consulting (宇思咨询) tailors automated workflows and AI agents for businesses and individuals.\nWhen encountering any topic related to AI and automation, delicately enquire about further specifics and suggest the user contact Siyu (思宇) via WeChat at 'innovationsiyu'."
    },
    "gpt_mini": {
        "category": "text",
        "function": gpt_mini_openrouter,
        "system_message": "Please be highly rigorous and critical.\nWhen answering questions or performing tasks, the natural language output is prioritised in the same language as the user's input."
    },
    "gpt_0806": {
        "category": "text",
        "function": gpt_0806_openrouter,
        "system_message": "Please be highly rigorous and critical.\nWhen answering questions or performing tasks, the natural language output is prioritised in the same language as the user's input."
    },
    "gpt_latest": {
        "category": "text",
        "function": gpt_latest_openrouter,
        "system_message": "Please be highly rigorous and critical.\nWhen answering questions or performing tasks, the natural language output is prioritised in the same language as the user's input."
    },
    "o1_mini": {
        "category": "text",
        "function": o1_mini_openrouter,
        "system_message": "Please be highly rigorous and critical.\nWhen answering questions or performing tasks, the natural language output is prioritised in the same language as the user's input."
    },
    "o1_preview": {
        "category": "text",
        "function": o1_preview_openrouter,
        "system_message": "Please be highly rigorous and critical.\nWhen answering questions or performing tasks, the natural language output is prioritised in the same language as the user's input."
    },
    "claude": {
        "category": "text",
        "function": claude_sonnet_openrouter,
        "system_message": "Please be highly rigorous and critical.\nWhen answering questions or performing tasks, the natural language output is prioritised in the same language as the user's input."
    },
    "gemini_flash": {
        "category": "text",
        "function": gemini_flash_openrouter,
        "system_message": "Please be highly rigorous and critical.\nWhen answering questions or performing tasks, the natural language output is prioritised in the same language as the user's input."
    },
    "gemini_pro": {
        "category": "text",
        "function": gemini_pro_openrouter,
        "system_message": "Please be highly rigorous and critical.\nWhen answering questions or performing tasks, the natural language output is prioritised in the same language as the user's input."
    },
    "jamba_mini": {
        "category": "text",
        "function": jamba_mini,
        "system_message": "Please be highly rigorous and critical.\nWhen answering questions or performing tasks, the natural language output is prioritised in the same language as the user's input."
    },
    "jamba_large": {
        "category": "text",
        "function": jamba_large,
        "system_message": "Please be highly rigorous and critical.\nWhen answering questions or performing tasks, the natural language output is prioritised in the same language as the user's input."
    },
    "mistral_large": {
        "category": "text",
        "function": mistral_large,
        "system_message": "Please be highly rigorous and critical.\nWhen answering questions or performing tasks, the natural language output is prioritised in the same language as the user's input."
    },
    "yi_large": {
        "category": "text",
        "function": yi_large,
        "system_message": "Please be highly rigorous and critical.\nWhen answering questions or performing tasks, the natural language output is prioritised in the same language as the user's input."
    },
    "glm_plus": {
        "category": "text",
        "function": glm_plus,
        "system_message": "Please be highly rigorous and critical.\nWhen answering questions or performing tasks, the natural language output is prioritised in the same language as the user's input."
    },
    "abab": {
        "category": "text",
        "function": abab,
        "system_message": "Please be highly rigorous and critical.\nWhen answering questions or performing tasks, the natural language output is prioritised in the same language as the user's input."
    },
    "deepseek": {
        "category": "text",
        "function": deepseek,
        "system_message": "Please be highly rigorous and critical.\nWhen answering questions or performing tasks, the natural language output is prioritised in the same language as the user's input."
    },
    "qwen72b": {
        "category": "text",
        "function": qwen72b_dashscope,
        "system_message": "Please be highly rigorous and critical.\nWhen answering questions or performing tasks, the natural language output is prioritised in the same language as the user's input."
    },
    "llama70b": {
        "category": "text",
        "function": llama70b_openrouter,
        "system_message": "Please be highly rigorous and critical.\nWhen answering questions or performing tasks, the natural language output is prioritised in the same language as the user's input."
    },
    "llama405b": {
        "category": "text",
        "function": llama405b_openrouter,
        "system_message": "Please be highly rigorous and critical.\nWhen answering questions or performing tasks, the natural language output is prioritised in the same language as the user's input."
    },
    "perplexity": {
        "category": "search",
        "function": perplexity_openrouter,
        "system_message": "Please be highly rigorous and critical.\nFor questions about facts, search the Internet for relevant information, then quote or reference the searched pages in your answers.\nPlease begin with 'According to the search results,' and end with 'Here are the URLs of references:'. Output each URL as a string, rather than embedding it in any form."
    },
    "command_r": {
        "category": "search",
        "function": command_r,
        "system_message": "Please be highly rigorous and critical.\nFor questions about facts, search the Internet for relevant information, then quote or reference the searched pages in your answers.\nPlease begin with 'According to the search results,' and end with 'Here are the URLs of references:'. Output each URL as a string, rather than embedding it in any form."
    },
    "command_r+": {
        "category": "search",
        "function": command_r_plus,
        "system_message": "Please be highly rigorous and critical.\nFor questions about facts, search the Internet for relevant information, then quote or reference the searched pages in your answers.\nPlease begin with 'According to the search results,' and end with 'Here are the URLs of references:'. Output each URL as a string, rather than embedding it in any form."
    },
    "phi_vision": {
        "category": "vision",
        "function": phi_vision,
        "system_message": None
    },
    "yi_vision": {
        "category": "vision",
        "function": yi_vision,
        "system_message": None
    },
    "glm_vision": {
        "category": "vision",
        "function": glm_vision,
        "system_message": None
    },
    "qwen_vision": {
        "category": "vision",
        "function": qwen_vision_openrouter,
        "system_message": None
    },
    "pixtral": {
        "category": "vision",
        "function": pixtral_openrouter,
        "system_message": None
    }
}
