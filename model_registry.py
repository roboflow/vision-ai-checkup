import os

from models.anthropic import AnthropicModel
from models.cohere import CohereModel
from models.custom_openai import CustomOpenAIModel
from models.ollama import OllamaModel
from models.gemini import GeminiModel
from models.openai import OpenAIModel
from models.smolvlm import SmolVLMModel

# Logo URL constants
QWEN_LOGO = "https://cdn-avatars.huggingface.co/v1/production/uploads/620760a26e3b7210c2ff1943/-s1gyJfvbE1RgO5iBeNOi.png"
OPENAI_LOGO = "https://openai.com/favicon.ico"
ANTHROPIC_LOGO = "https://www.anthropic.com/favicon.ico"
GOOGLE_LOGO = "https://www.google.com/favicon.ico"
MISTRAL_LOGO = "https://mistral.ai/favicon.ico"
META_LOGO = "https://signsalad.com/wp-content/uploads/2021/11/Screenshot-2021-11-03-at-12.14.11.png"
COHERE_LOGO = "https://cohere.com/favicon.ico"
NVIDIA_LOGO = "https://www.nvidia.com/favicon.ico"

MODEL_REGISTRY = {
    # --- OpenAI ---
    "GPT-4.1": {
        "open_weights": False,
        "weights_size": "",
        "logo": OPENAI_LOGO,
        "playground_slug": "",
        "provider": lambda: OpenAIModel(model_id="gpt-4.1"),
    },
    "GPT-4.1 Mini": {
        "open_weights": False,
        "weights_size": "",
        "logo": OPENAI_LOGO,
        "playground_slug": "",
        "provider": lambda: OpenAIModel(model_id="gpt-4.1-mini"),
    },
    "GPT-4.1 Nano": {
        "open_weights": False,
        "weights_size": "",
        "logo": OPENAI_LOGO,
        "playground_slug": "",
        "provider": lambda: OpenAIModel(model_id="gpt-4.1-nano"),
    },
    "ChatGPT-4o": {
        "open_weights": False,
        "weights_size": "",
        "logo": OPENAI_LOGO,
        "playground_slug": "",
        "provider": None,
    },
    "ChatGPT-4o (Medium Reasoning)": {
        "open_weights": False,
        "weights_size": "",
        "logo": OPENAI_LOGO,
        "playground_slug": "",
        "provider": lambda: OpenAIModel(model_id="chatgpt-4o-latest"),
    },
    "GPT-5.4": {
        "open_weights": False,
        "weights_size": "",
        "logo": OPENAI_LOGO,
        "playground_slug": "openai/gpt-5-2",
        "provider": lambda: OpenAIModel(model_id="gpt-5.4-2026-03-05"),
    },
    "GPT-5 Mini": {
        "open_weights": False,
        "weights_size": "",
        "logo": OPENAI_LOGO,
        "playground_slug": "openai/gpt-5-mini",
        "provider": lambda: OpenAIModel(model_id="gpt-5-mini"),
    },
    "GPT-5 Nano": {
        "open_weights": False,
        "weights_size": "",
        "logo": OPENAI_LOGO,
        "playground_slug": "openai/gpt-5-nano",
        "provider": lambda: OpenAIModel(model_id="gpt-5-nano"),
    },
    "GPT-5 Chat": {
        "open_weights": False,
        "weights_size": "",
        "logo": OPENAI_LOGO,
        "playground_slug": "openai/gpt-5",
        "provider": None,
    },
    "OpenAI O1": {
        "open_weights": False,
        "weights_size": "",
        "logo": OPENAI_LOGO,
        "playground_slug": "",
        "provider": lambda: OpenAIModel(model_id="o1"),
    },
    "OpenAI O1 Pro": {
        "open_weights": False,
        "weights_size": "",
        "logo": OPENAI_LOGO,
        "playground_slug": "",
        "provider": None,
    },
    "OpenAI O3": {
        "open_weights": False,
        "weights_size": "",
        "logo": OPENAI_LOGO,
        "playground_slug": "",
        "provider": None,
    },
    "OpenAI O3 Mini": {
        "open_weights": False,
        "weights_size": "",
        "logo": OPENAI_LOGO,
        "playground_slug": "",
        "provider": None,
    },
    "OpenAI O4 Mini": {
        "open_weights": False,
        "weights_size": "",
        "logo": OPENAI_LOGO,
        "playground_slug": "",
        "provider": None,
    },
    "OpenAI O4 Mini (Medium Reasoning)": {
        "open_weights": False,
        "weights_size": "",
        "logo": OPENAI_LOGO,
        "playground_slug": "",
        "provider": lambda: OpenAIModel(model_id="o4-mini"),
    },
    "OpenAI o3-pro": {
        "open_weights": False,
        "weights_size": "",
        "logo": OPENAI_LOGO,
        "playground_slug": "",
        "provider": None,
    },
    # --- Anthropic ---
    "Claude 3.7 Sonnet": {
        "open_weights": False,
        "weights_size": "",
        "logo": ANTHROPIC_LOGO,
        "playground_slug": "",
        "provider": lambda: AnthropicModel(model_id="claude-3-7-sonnet-20250219"),
    },
    "Claude 3.5 Haiku": {
        "open_weights": False,
        "weights_size": "",
        "logo": ANTHROPIC_LOGO,
        "playground_slug": "anthropic/claude-4-5-haiku",
        "provider": lambda: AnthropicModel(model_id="claude-3-5-haiku-20241022"),
    },
    "Claude 4 Sonnet": {
        "open_weights": False,
        "weights_size": "",
        "logo": ANTHROPIC_LOGO,
        "playground_slug": "anthropic/claude-4-sonnet",
        "provider": lambda: AnthropicModel(model_id="claude-sonnet-4-20250514"),
    },
    "Claude 4 Opus": {
        "open_weights": False,
        "weights_size": "",
        "logo": ANTHROPIC_LOGO,
        "playground_slug": "anthropic/claude-4-opus",
        "provider": lambda: AnthropicModel(model_id="claude-opus-4-20250514"),
    },
    "Claude 4.6 Opus": {
        "open_weights": False,
        "weights_size": "",
        "logo": ANTHROPIC_LOGO,
        "playground_slug": "anthropic/claude-opus-4-6",
        "provider": lambda: AnthropicModel(model_id="claude-opus-4-6", thinking_budget=16000),
    },
    "Claude 4.1 Opus": {
        "open_weights": False,
        "weights_size": "",
        "logo": ANTHROPIC_LOGO,
        "playground_slug": "anthropic/claude-4-1-opus",
        "provider": lambda: AnthropicModel(model_id="claude-opus-4-1-20250805"),
    },
    # --- Google ---
    "Gemini 2.5 Pro Preview": {
        "open_weights": False,
        "weights_size": "",
        "logo": GOOGLE_LOGO,
        "playground_slug": "",
        "provider": None,
    },
    "Gemini 2.5 Pro": {
        "open_weights": False,
        "weights_size": "",
        "logo": GOOGLE_LOGO,
        "playground_slug": "google/gemini-2-5-pro",
        "provider": lambda: GeminiModel(model_id="gemini-2.5-pro"),
    },
    "Gemini 2.0 Flash": {
        "open_weights": False,
        "weights_size": "",
        "logo": GOOGLE_LOGO,
        "playground_slug": "",
        "provider": lambda: GeminiModel(model_id="gemini-2.0-flash"),
    },
    "Gemini 2.0 Flash Lite": {
        "open_weights": False,
        "weights_size": "",
        "logo": GOOGLE_LOGO,
        "playground_slug": "",
        "provider": lambda: GeminiModel(model_id="gemini-2.0-flash-lite"),
    },
    "Gemini 2.5 Flash Preview": {
        "open_weights": False,
        "weights_size": "",
        "logo": GOOGLE_LOGO,
        "playground_slug": "",
        "provider": None,
    },
    "Gemini 2.5 Flash": {
        "open_weights": False,
        "weights_size": "",
        "logo": GOOGLE_LOGO,
        "playground_slug": "google/gemini-2-5-flash",
        "provider": lambda: GeminiModel(model_id="gemini-2.5-flash"),
    },
    "Gemini 2.5 Flash Lite": {
        "open_weights": False,
        "weights_size": "",
        "logo": GOOGLE_LOGO,
        "playground_slug": "google/gemini-2-5-flash-lite",
        "provider": lambda: GeminiModel(model_id="gemini-2.5-flash-lite"),
    },
    "Gemini 2.5 Flash-Lite Preview": {
        "open_weights": False,
        "weights_size": "",
        "logo": GOOGLE_LOGO,
        "playground_slug": "",
        "provider": None,
    },
    "Gemini 3 Flash": {
        "open_weights": False,
        "weights_size": "",
        "logo": GOOGLE_LOGO,
        "playground_slug": "google/gemini-3-flash",
        "provider": lambda: GeminiModel(model_id="gemini-3-flash-preview"),
    },
    "Gemini 3 Flash (Tools)": {
        "open_weights": False,
        "weights_size": "",
        "logo": GOOGLE_LOGO,
        "playground_slug": "google/gemini-3-flash",
        "provider": lambda: GeminiModel(model_id="gemini-3-flash-preview", enable_code_execution=True),
    },
    "Gemini 3 Pro Preview": {
        "open_weights": False,
        "weights_size": "",
        "logo": GOOGLE_LOGO,
        "playground_slug": "google/gemini-3-pro",
        "provider": None,
    },
    "Gemini 3.1 Pro": {
        "open_weights": False,
        "weights_size": "",
        "logo": GOOGLE_LOGO,
        "playground_slug": "google/gemini-3-1-pro",
        "provider": lambda: GeminiModel(model_id="gemini-3.1-pro-preview"),
    },
    "Gemini 3.1 Pro (Tools)": {
        "open_weights": False,
        "weights_size": "",
        "logo": GOOGLE_LOGO,
        "playground_slug": "google/gemini-3-1-pro",
        "provider": lambda: GeminiModel(model_id="gemini-3.1-pro-preview", enable_code_execution=True),
    },
    "Gemma 3 1B": {
        "open_weights": True,
        "weights_size": "1B",
        "logo": MISTRAL_LOGO,
        "playground_slug": "",
        "provider": None,
    },
    "Gemma 3 4B": {
        "open_weights": True,
        "weights_size": "4B",
        "logo": GOOGLE_LOGO,
        "playground_slug": "google/gemma-3-4b",
        "provider": lambda: CustomOpenAIModel(
            model_id="google/gemma-3-4b-it",
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY"),
        ),
    },
    "Gemma 3 27b": {
        "open_weights": True,
        "weights_size": "27B",
        "logo": GOOGLE_LOGO,
        "playground_slug": "google/gemma-3-27b",
        "provider": lambda: CustomOpenAIModel(
            model_id="google/gemma-3-27b-it:nebius",
            base_url="https://router.huggingface.co/v1",
            api_key=os.environ.get("HUGGINGFACE_API_KEY"),
        ),
    },
    "Gemma 3n 4B": {
        "open_weights": True,
        "weights_size": "4B",
        "logo": GOOGLE_LOGO,
        "playground_slug": "",
        "provider": None,
    },
    # --- Meta ---
    "Llama 3 11B Vision": {
        "open_weights": True,
        "weights_size": "11B",
        "logo": META_LOGO,
        "playground_slug": "meta/llama-3-2-vision-11b",
        "provider": None,
    },
    "Llama 4 Scout 17B": {
        "open_weights": True,
        "weights_size": "109B/A17B",
        "logo": META_LOGO,
        "playground_slug": "meta/llama-4-scout",
        "provider": lambda: CustomOpenAIModel(
            model_id="meta-llama/Llama-4-Scout-17B-16E-Instruct:together",
            base_url="https://router.huggingface.co/v1",
            api_key=os.environ.get("HUGGINGFACE_API_KEY"),
        ),
    },
    "Llama 4 Maverick 17B": {
        "open_weights": True,
        "weights_size": "401B/A17B",
        "logo": META_LOGO,
        "playground_slug": "meta/llama-4-maverick",
        "provider": lambda: CustomOpenAIModel(
            model_id="meta-llama/llama-4-maverick",
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY"),
        ),
    },
    # --- Mistral ---
    "Mistral Medium 3": {
        "open_weights": True,
        "weights_size": "73B",
        "logo": MISTRAL_LOGO,
        "playground_slug": "mistral/mistral-medium-3-1",
        "provider": lambda: CustomOpenAIModel(
            model_id="mistralai/mistral-medium-3",
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY"),
        ),
    },
    "Mistral Small 3.1 24B": {
        "open_weights": True,
        "weights_size": "24B",
        "logo": MISTRAL_LOGO,
        "playground_slug": "mistral/mistral-small-3-1-24b",
        "provider": lambda: CustomOpenAIModel(
            model_id="mistralai/mistral-small-3.1-24b-instruct",
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY"),
        ),
    },
    "Mistral Small 3.1 24b": {
        "open_weights": True,
        "weights_size": "24B",
        "logo": MISTRAL_LOGO,
        "playground_slug": "mistral/mistral-small-3-1-24b",
        "provider": None,
    },
    # --- Cohere ---
    "Cohere Aya Vision 8B": {
        "open_weights": True,
        "weights_size": "8B",
        "logo": COHERE_LOGO,
        "playground_slug": "",
        "provider": lambda: CohereModel(model_id="c4ai-aya-vision-8b"),
    },
    "Cohere Aya Vision 32B": {
        "open_weights": True,
        "weights_size": "32B",
        "logo": COHERE_LOGO,
        "playground_slug": "",
        "provider": lambda: CohereModel(model_id="c4ai-aya-vision-32b"),
    },
    # --- Qwen ---
    "Qwen 2.5 VL 7B": {
        "open_weights": True,
        "weights_size": "7B",
        "logo": QWEN_LOGO,
        "playground_slug": "qwen/qwen2-5-vl-7b-instruct",
        "provider": lambda: CustomOpenAIModel(
            model_id="Qwen/Qwen2.5-VL-7B-Instruct:hyperbolic",
            base_url="https://router.huggingface.co/v1",
            api_key=os.environ.get("HUGGINGFACE_API_KEY"),
        ),
    },
    "Qwen 3.5 Plus": {
        "open_weights": True,
        "weights_size": "397B/A17B",
        "logo": QWEN_LOGO,
        "playground_slug": "qwen/qwen3-vl-235b-a22b-instruct",
        "provider": lambda: CustomOpenAIModel(
            model_id="qwen/qwen3.5-plus-02-15",
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY"),
        ),
    },
    "Qwen 3.5 397B": {
        "open_weights": True,
        "weights_size": "397B/A74B",
        "logo": QWEN_LOGO,
        "playground_slug": "qwen/qwen3-5-397b-a17b",
        "provider": lambda: CustomOpenAIModel(
            model_id="qwen/qwen3.5-397b-a17b",
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY"),
        ),
    },
    "Qwen 3.5 122B (A10B)": {
        "open_weights": True,
        "weights_size": "122B/A10B",
        "logo": QWEN_LOGO,
        "playground_slug": "",
        "provider": lambda: CustomOpenAIModel(
            model_id="qwen/qwen3.5-122b-a10b",
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY"),
        ),
    },
    "Qwen 3.5 35B (A3B)": {
        "open_weights": True,
        "weights_size": "35B/A3B",
        "logo": QWEN_LOGO,
        "playground_slug": "qwen/qwen3-vl-30b-a3b-instruct",
        "provider": lambda: CustomOpenAIModel(
            model_id="qwen/qwen3.5-35b-a3b",
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY"),
        ),
    },
    "Qwen 3.5 27B": {
        "open_weights": True,
        "weights_size": "27B",
        "logo": QWEN_LOGO,
        "playground_slug": "qwen/qwen-vl-max",
        "provider": lambda: CustomOpenAIModel(
            model_id="qwen/qwen3.5-27b",
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY"),
        ),
    },
    "Qwen 3.5 0.8B": {
        "open_weights": True,
        "weights_size": "0.8B",
        "logo": QWEN_LOGO,
        "playground_slug": "",
        "provider": lambda: OllamaModel(
            model_id="qwen3.5:0.8b",
            base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
        ),
    },
    "Qwen 3.5 2B": {
        "open_weights": True,
        "weights_size": "2B",
        "logo": QWEN_LOGO,
        "playground_slug": "",
        "provider": lambda: OllamaModel(
            model_id="qwen3.5:2b",
            base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
        ),
    },
    "Qwen 3.5 4B": {
        "open_weights": True,
        "weights_size": "4B",
        "logo": QWEN_LOGO,
        "playground_slug": "",
        "provider": lambda: OllamaModel(
            model_id="qwen3.5:4b",
            base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
        ),
    },
    "Qwen 3.5 9B": {
        "open_weights": True,
        "weights_size": "9B",
        "logo": QWEN_LOGO,
        "playground_slug": "qwen/qwen3-vl-8b-instruct",
        "provider": lambda: OllamaModel(
            model_id="qwen3.5:9b",
            base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
        ),
    },
    # --- Microsoft ---
    "Phi 4 Multimodal": {
        "open_weights": True,
        "weights_size": "5.6B",
        "logo": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/44/Microsoft_logo.svg/1024px-Microsoft_logo.svg.png?20210729021049",
        "playground_slug": "",
        "provider": lambda: CustomOpenAIModel(
            model_id="microsoft/phi-4-multimodal-instruct",
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY"),
        ),
    },
    # --- xAI ---
    "Grok 4": {
        "open_weights": False,
        "weights_size": "",
        "logo": "/images/xai-icon.svg",
        "playground_slug": "xai/grok-4",
        "provider": lambda: CustomOpenAIModel(
            model_id="x-ai/grok-4",
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY"),
        ),
    },
    "Grok 4.1 Fast": {
        "open_weights": False,
        "weights_size": "",
        "logo": "/images/xai-icon.svg",
        "playground_slug": "",
        "provider": lambda: CustomOpenAIModel(
            model_id="x-ai/grok-4.1-fast",
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY"),
        ),
    },
    # --- GLM / Kimi ---
    "GLM 4.6v": {
        "open_weights": True,
        "weights_size": "106B",
        "logo": "/images/z-icon.svg",
        "playground_slug": "",
        "provider": lambda: CustomOpenAIModel(
            model_id="z-ai/glm-4.6v",
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY"),
        ),
    },
    "Kimi k2.5": {
        "open_weights": True,
        "weights_size": "1T/A32B",
        "logo": "/images/kimi-icon.ico",
        "playground_slug": "",
        "provider": lambda: CustomOpenAIModel(
            model_id="moonshotai/kimi-k2.5",
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY"),
        ),
    },
    # --- Allen AI ---
    "Molmo2 8B": {
        "open_weights": True,
        "weights_size": "8B",
        "logo": "https://allenai.org/favicon.ico",
        "playground_slug": "",
        "provider": lambda: CustomOpenAIModel(
            model_id="allenai/molmo-2-8b",
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY"),
        ),
    },
    # --- NVIDIA ---
    "Cosmos Reason2 2B": {
        "open_weights": True,
        "weights_size": "2B",
        "logo": NVIDIA_LOGO,
        "playground_slug": "",
        "provider": lambda: CustomOpenAIModel(
            model_id="nvidia/Cosmos-Reason2-2B",
            base_url=os.environ.get("VLLM_BASE_URL", "http://localhost:8000") + "/v1",
            api_key="no-auth",
        ),
    },
    "Cosmos Reason2 8B": {
        "open_weights": True,
        "weights_size": "8B",
        "logo": NVIDIA_LOGO,
        "playground_slug": "",
        "provider": lambda: CustomOpenAIModel(
            model_id="nvidia/Cosmos-Reason2-8B",
            base_url=os.environ.get("VLLM_BASE_URL", "http://localhost:8000") + "/v1",
            api_key="no-auth",
        ),
    },
    # --- HuggingFace ---
    "SmolVLM2 2.2B": {
        "open_weights": True,
        "weights_size": "2.2B",
        "logo": "https://huggingface.co/favicon.ico",
        "playground_slug": "",
        "provider": lambda: SmolVLMModel(
            base_url=os.environ.get("SMOLVLM_BASE_URL", "http://localhost:5000"),
        ),
    },
    # --- Reka ---
    "Reka Edge": {
        "open_weights": True,
        "weights_size": "7B",
        "logo": "https://framerusercontent.com/images/XekAjHcWv2prUE9pJiSj61CD54.jpg",
        "playground_slug": "",
        "provider": lambda: CustomOpenAIModel(
            model_id="reka-edge-2603",
            base_url="https://api.reka.ai/v1",
            api_key=os.environ.get("REKA_KEY"),
            supports_response_format=False,
        ),
    },
    # --- Arcee ---
    "Arcee.ai Spotlight": {
        "open_weights": False,
        "weights_size": "",
        "logo": "https://cdn.prod.website-files.com/6781a10424493fe352bc6cb5/678e92cb5d392e76c953e690_Favicon.png",
        "playground_slug": "",
        "provider": lambda: CustomOpenAIModel(
            model_id="arcee-ai/spotlight",
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY"),
        ),
    },
}
