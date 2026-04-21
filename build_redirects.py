"""Build visioncheckup.com -> playground.roboflow.com redirect map AND
overwrite every generated HTML page in docs/ with a redirect page.

Sources:
- Per-model pages: `playground_slug` field in model_registry.py,
  merged with MANUAL_OVERRIDES below for models that don't have one.
  Target URL pattern: https://playground.roboflow.com/models/{playground_slug}
- Category pages (/annotation-understanding.html etc): CATEGORY_TASKS below,
  mapped to https://playground.roboflow.com/evals?task=<Label>.
- Everything else: https://playground.roboflow.com/evals

Run after build_site.py:
    python build_site.py && python build_redirects.py
"""
import csv
import json
import os

from jinja2 import Environment, FileSystemLoader

from model_registry import MODEL_REGISTRY


OUTPUT_DIR = "docs"
TEMPLATES_DIR = "templates"


PLAYGROUND_BASE = "https://playground.roboflow.com"
EVALS_FALLBACK = f"{PLAYGROUND_BASE}/evals"
MODELS_FALLBACK = EVALS_FALLBACK


def slugify(value):
    value = value.lower().replace(" ", "-")
    return "".join(c if c.isalnum() or c == "-" else "-" for c in value)


# Manual per-model overrides for entries whose registry playground_slug is "".
# Values are the path after /models/ (same shape as playground_slug in registry),
# or "" to send the page to the /models fallback.
# Filled from WebSearch results on playground.roboflow.com.
# Corrections that WIN over a non-empty playground_slug already in the registry.
# Use when the registry value points to a slug that doesn't exist on playground.
# Confirmed via WebSearch hits on playground.roboflow.com.
REGISTRY_CORRECTIONS = {
    # Registry had these pointing at qwen3-vl-* slugs, but playground uses qwen3-5-* names.
    "Qwen 3.5 Plus": "qwen/qwen3-5-plus",         # was qwen/qwen3-vl-235b-a22b-instruct (unverified)
    "Qwen 3.5 35B (A3B)": "qwen/qwen3-5-35b-a3b", # was qwen/qwen3-vl-30b-a3b-instruct
    "Qwen 3.5 9B": "qwen/qwen3-5-9b",             # was qwen/qwen3-vl-8b-instruct
}


MANUAL_OVERRIDES = {
    # --- OpenAI ---
    "GPT-4.1": "openai/gpt-4-1",
    "GPT-4.1 Mini": "openai/gpt-4-1-mini",
    "GPT-4.1 Nano": "openai/gpt-4-1-nano",
    "ChatGPT-4o": "openai/gpt-4o",
    "ChatGPT-4o (Medium Reasoning)": "openai/gpt-4o",
    "OpenAI O1": "openai/o1",
    "OpenAI O1 Pro": "openai/o1-pro",
    "OpenAI O3": "openai/o3",
    "OpenAI O3 Mini": "openai/o3-mini",
    "OpenAI O4 Mini": "openai/o4-mini",
    "OpenAI O4 Mini (Medium Reasoning)": "openai/o4-mini",
    "OpenAI o3-pro": "openai/o3-pro",
    # --- Anthropic ---
    "Claude 3.7 Sonnet": "anthropic/claude-3-7-sonnet",
    # --- Google ---
    "Gemini 2.5 Pro Preview": "google/gemini-2-5-pro",
    "Gemini 2.0 Flash": "google/gemini-2-0-flash",
    "Gemini 2.0 Flash Lite": "google/gemini-2-0-flash-lite",
    "Gemini 2.5 Flash Preview": "google/gemini-2-5-flash",
    "Gemini 2.5 Flash-Lite Preview": "google/gemini-2-5-flash-lite",
    "Gemma 3 1B": "",
    "Gemma 3n 4B": "",
    "Gemma 4 26B (A4B)": "google/gemma-4-26b-a4b",
    "Gemma 4 31B": "google/gemma-4-31b",
    # --- Cohere ---
    "Cohere Aya Vision 8B": "cohere/aya-vision-8b",
    "Cohere Aya Vision 32B": "cohere/aya-vision-32b",
    # --- Qwen ---
    "Qwen 3.6 Plus": "qwen/qwen3-6-plus",
    "Qwen 3.5 122B (A10B)": "",
    "Qwen 3.5 0.8B": "",
    "Qwen 3.5 2B": "",
    "Qwen 3.5 4B": "",
    # --- Others ---
    "Phi 4 Multimodal": "microsoft/phi-4-multimodal",
    "Grok 4.1 Fast": "xai/grok-4-1-fast",
    "GLM 4.6v": "",
    "GLM-OCR": "",
    "Kimi k2.5": "moonshot/kimi-k2-5",
    "Molmo2 8B": "allenai/molmo-2-8b",
    "Cosmos Reason2 2B": "",
    "Cosmos Reason2 8B": "",
    "SmolVLM2 2.2B": "huggingface/smolvlm2-2-2b",
    "Reka Edge": "reka/reka-edge",
    "LFM 2 24B (A2B)": "",
    "LFM 2.5 VL 1.6B": "liquid/lfm-2-5-vl-1-6b",
    "Arcee.ai Spotlight": "arcee/spotlight",
}


# Vision-checkup category page -> playground /evals?task=<name>.
# Task name uses the vision-checkup category label verbatim (URL-encoded).
CATEGORY_TASKS = [
    "Annotation Understanding",
    "CAPTCHA",
    "Color Identification",
    "Defect Detection",
    "Document Understanding",
    "Localization",
    "Object Counting",
    "Object Detection",
    "Object Measurement",
    "Object Understanding",
    "OCR",
    "Receipt Reading",
    "Sign Understanding",
    "Spatial Relations",
    "Web Action Understanding",
]


def slug_from_category(label):
    return label.lower().replace(" ", "-").replace("+", "-")


def build_category_redirects():
    from urllib.parse import quote_plus
    redirects = {}
    for label in CATEGORY_TASKS:
        path = f"/{slug_from_category(label)}.html"
        redirects[path] = f"{PLAYGROUND_BASE}/evals?task={quote_plus(label)}"
    return redirects


def build_model_redirects():
    redirects = {}
    for model_name, meta in MODEL_REGISTRY.items():
        vc_path = f"/{slugify(model_name)}/"
        slug = (
            REGISTRY_CORRECTIONS.get(model_name)
            or meta.get("playground_slug")
            or MANUAL_OVERRIDES.get(model_name, "")
        )
        if slug:
            redirects[vc_path] = f"{PLAYGROUND_BASE}/models/{slug}"
        else:
            redirects[vc_path] = MODELS_FALLBACK
    return redirects


def file_to_web_path(rel_path):
    """Map a docs/**/*.html path to the lookup key shape used in redirects.json.

    Examples:
        ocr.html                         -> /ocr.html
        claude-4-opus/index.html         -> /claude-4-opus/
        compare/gpt-4-vs-gpt-5/index.html -> /compare/gpt-4-vs-gpt-5/
        index.html                       -> /
    """
    rel = rel_path.replace(os.sep, "/")
    if rel.endswith("/index.html"):
        return "/" + rel[: -len("index.html")]
    if rel == "index.html":
        return "/"
    return "/" + rel


def resolve_target(web_path, model_redirects, category_redirects):
    if web_path in model_redirects:
        return model_redirects[web_path]
    if web_path in category_redirects:
        return category_redirects[web_path]
    return EVALS_FALLBACK


def write_redirect_pages(model_redirects, category_redirects):
    env = Environment(loader=FileSystemLoader(TEMPLATES_DIR))
    template = env.get_template("redirect.html")

    counts = {"model": 0, "category": 0, "fallback": 0}
    total = 0

    for root, _, files in os.walk(OUTPUT_DIR):
        for name in files:
            if not name.endswith(".html"):
                continue
            abs_path = os.path.join(root, name)
            rel = os.path.relpath(abs_path, OUTPUT_DIR)
            web_path = file_to_web_path(rel)
            target = resolve_target(web_path, model_redirects, category_redirects)
            html = template.render(target=target, target_json=json.dumps(target))
            with open(abs_path, "w") as f:
                f.write(html)
            total += 1
            if web_path in model_redirects:
                counts["model"] += 1
            elif web_path in category_redirects:
                counts["category"] += 1
            else:
                counts["fallback"] += 1

    return total, counts


def main():
    model_redirects = build_model_redirects()
    category_redirects = build_category_redirects()

    out = {
        "fallback": EVALS_FALLBACK,
        "models": dict(sorted(model_redirects.items())),
        "categories": dict(sorted(category_redirects.items())),
    }
    out_path = os.path.join(os.path.dirname(__file__), "redirects.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    mapped_models = sum(
        1 for v in model_redirects.values() if v != MODELS_FALLBACK
    )
    total_models = len(model_redirects)
    print(f"Wrote {out_path}")
    print(f"  Model redirects: {mapped_models}/{total_models} mapped ({total_models - mapped_models} fall back)")
    print(f"  Category redirects: {len(category_redirects)}")

    if os.path.isdir(OUTPUT_DIR):
        total, counts = write_redirect_pages(model_redirects, category_redirects)
        print(f"Overwrote {total} HTML pages in {OUTPUT_DIR}/ with redirect shim")
        print(f"  matched model page:    {counts['model']}")
        print(f"  matched category page: {counts['category']}")
        print(f"  fell back to /evals:   {counts['fallback']}")
    else:
        print(f"Skipped HTML overwrite: {OUTPUT_DIR}/ not found. Run build_site.py first.")


if __name__ == "__main__":
    main()
