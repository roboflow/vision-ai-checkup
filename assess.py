import concurrent.futures
import csv
import orjson
import os
import shutil
import json
import string
import time
from collections import defaultdict
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time
from urllib.parse import quote_plus

from jinja2 import Environment, FileSystemLoader
from tqdm import tqdm
from itertools import combinations
import optparse

is_in_incremental_mode = False
def parse_args():
    parser = optparse.OptionParser()
    parser.add_option(
        "--incremental",
        action="store_true",
        dest="incremental",
        default=False,
        help="Run in incremental mode, only updating changed files.",
    )
    options, _ = parser.parse_args()
    return options

options = parse_args()

if options.incremental:
    is_in_incremental_mode = True
    print("Running in incremental mode. Only changed files will be updated.")

if os.path.exists("docs"):
    shutil.rmtree("docs")

OUTPUT_DIR = "docs"
BASE_IMAGE_DIR = "images/"
CONCURRENCY_OVERRIDE_MODELS = ["Llama 3.1"]
from models.anthropic import AnthropicModel
from models.cohere import CohereModel
from models.custom_openai import CustomOpenAIModel
from models.gemini import GeminiModel
from models.openai import OpenAIModel
# from models.open_router import OpenRouterModel
# from models.roboflow_workflow import RoboflowWorkflow
# from utils.data_types import BoundingBoxes

open_or_closed_source = {
    "OpenAI O4 Mini": "closed",
    "GPT-4.1": "closed",
    "ChatGPT-4o": "closed",
    "GPT-4.1 Mini": "closed",
    "GPT-4.1 Nano": "closed",
    "OpenAI O1": "closed",
    "Claude 3.7 Sonnet": "closed",
    "Claude 3.5 Haiku": "closed",
    "Gemini 2.5 Pro Preview": "closed",
    "Gemini 2.0 Flash": "closed",
    "Gemini 2.0 Flash Lite": "closed",
    "Gemini 2.5 Flash Preview": "closed",
    "Cohere Aya Vision 8B": "closed",
    "Cohere Aya Vision 32B": "closed",
    "Qwen 2.5 VL 7B": "open",
    "Llama 4 Scout 17B": "closed",
    "Llama 3 11B Vision": "closed",
    "Gemma 3 27b": "closed",
    "Mistral Medium 3": "open",
    "Gemma 3 1B": "open",
    "Mistral Small 3.1 24B": "open",
    "Gemma 3 4B": "open",
    "Phi 4 Multimodal": "closed",
    "Gemini 1.5 Flash": "closed",
    "Gemini 1.5 Pro": "closed",
    "Llama 4 Maverick 17B": "closed",
    "Claude 4 Opus": "closed",
    "OpenAI O3": "closed",
}

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "prompts"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "assessments"), exist_ok=True)


    env = Environment(loader=FileSystemLoader("templates"))
    template = env.get_template("index.html")
    card_template = env.get_template("card.html")
    prompts_template = env.get_template("prompts.html")
    assessment_template = env.get_template("assessment.html")
    compare_template = env.get_template("compare.html")
    sitemap_template = env.get_template("sitemap.xml")

    def slugify(value):
        """Convert a string to a slug."""
        value = value.lower()
        value = value.replace(" ", "-")
        value = "".join(
            c if c.isalnum() or c == "-" else "-" for c in value
        )  # replace non-alphanumeric chars with hyphen
        return value

    logos = {
        "Llama 3 11B Vision": "https://signsalad.com/wp-content/uploads/2021/11/Screenshot-2021-11-03-at-12.14.11.png",
        "Llama 3 70B": "https://signsalad.com/wp-content/uploads/2021/11/Screenshot-2021-11-03-at-12.14.11.png",
        "Llama 4 Maverick 17B": "https://signsalad.com/wp-content/uploads/2021/11/Screenshot-2021-11-03-at-12.14.11.png",
        "Llama 4 Scout 17B": "https://signsalad.com/wp-content/uploads/2021/11/Screenshot-2021-11-03-at-12.14.11.png",
        "GPT-4.1": "https://openai.com/favicon.ico",
        "ChatGPT-4o": "https://openai.com/favicon.ico",
        "OpenAI O1 Pro": "https://openai.com/favicon.ico",
        "GPT-4.1 Mini": "https://openai.com/favicon.ico",
        "GPT-4.1 Nano": "https://openai.com/favicon.ico",
        "OpenAI O3": "https://openai.com/favicon.ico",
        "OpenAI O3 Mini": "https://openai.com/favicon.ico",
        "OpenAI O1": "https://openai.com/favicon.ico",
        "Claude 3.7 Sonnet": "https://www.anthropic.com/favicon.ico",
        "Qwen 2.5 VL 7B": "https://cdn-avatars.huggingface.co/v1/production/uploads/620760a26e3b7210c2ff1943/-s1gyJfvbE1RgO5iBeNOi.png",
        "Claude 3.5 Haiku": "https://www.anthropic.com/favicon.ico",
        "Gemini 2.5 Pro Preview": "https://www.google.com/favicon.ico",
        "Gemini 2.0 Flash": "https://www.google.com/favicon.ico",
        "Gemini 2.0 Flash Lite": "https://www.google.com/favicon.ico",
        "Gemma 3 27b": "https://www.google.com/favicon.ico",
        "OpenAI O4 Mini": "https://openai.com/favicon.ico",
        "Gemini 2.5 Flash Preview": "https://www.google.com/favicon.ico",
        "Cohere Aya Vision 8B": "https://cohere.com/favicon.ico",
        "Cohere Aya Vision 32B": "https://cohere.com/favicon.ico",
        "Claude 4 Sonnet": "https://www.anthropic.com/favicon.ico",
        "Claude 4 Opus": "https://www.anthropic.com/favicon.ico",
        "Gemma 3n 4B": "https://www.google.com/favicon.ico",
        "Mistral Medium 3": "https://mistral.ai/favicon.ico",
        "Gemma 3 1B": "https://mistral.ai/favicon.ico",
        "Mistral Small 3.1 24b": "https://mistral.ai/favicon.ico",
        "Mistral Small 3.1 24B": "https://mistral.ai/favicon.ico",
        "Gemma 3 4B": "https://www.google.com/favicon.ico",
        "Gemini 1.5 Flash": "https://www.google.com/favicon.ico",
        "Gemini 1.5 Pro": "https://www.google.com/favicon.ico",
        "Arcee.ai Spotlight": "https://cdn.prod.website-files.com/6781a10424493fe352bc6cb5/678e92cb5d392e76c953e690_Favicon.png",
        "Phi 4 Multimodal": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/44/Microsoft_logo.svg/1024px-Microsoft_logo.svg.png?20210729021049",
    }

    def normalise_output(output):
        if not output:
            return ""

        output = output.strip().lower()
        output = output.translate(str.maketrans("", "", string.punctuation))

        return output.strip().replace(" ", "")


    with open("prompts.csv", "r") as file:
        reader = csv.DictReader(file)
        assessments = list(reader)
        for assessment in assessments:
            assessment["slug"] = slugify(assessment["assessment_name"])

    assessments_by_model = defaultdict(lambda: defaultdict(list))

    def run_model_with_prompt(model_name, model, assessment):
        print(model_name, assessment)
        if isinstance(assessment, str):
            # get assessment by file name
            assessment = [a for a in assessments if a["file_name"] == assessment][0]

        with open(
            os.path.join(BASE_IMAGE_DIR, assessment["file_name"]), "rb"
        ) as image_file:
            assessment["image"] = image_file.read()
        start_time = time.time()
        print(
            f"Running {model_name} with image {assessment['file_name']} and prompt {assessment['prompt']}"
        )

        result = model.run_with_retry(
            assessment["image"],
            assessment["prompt"],
            image_name=os.path.join(BASE_IMAGE_DIR, assessment["file_name"]),
        )
        # if result is none, try on compressed
        if result is None:
            print(f"Retrying {model_name} with compressed image")
            with open(
                os.path.join(BASE_IMAGE_DIR, "compressed/", assessment["file_name"].replace(".png", ".jpeg")), "rb"
            ) as image_file:
                assessment["image"] = image_file.read()
            result = model.run_with_retry(
                assessment["image"],
                assessment["prompt"],
                image_name=os.path.join(BASE_IMAGE_DIR, "compressed/", assessment["file_name"].replace(".png", ".jpeg")),
            )

        end_time = time.time()
        assessment["image"] = None

        answer = assessment["answer"]

        time_taken = end_time - start_time

        return model_name, assessment, result, answer, time_taken


    times_by_model = defaultdict(list)

    # if model_results.json exists, load it instead of running the models again
    if os.path.exists("./model_results.json") and not is_in_incremental_mode:
        with open("./model_results.json", "r") as file:
            final_results = orjson.loads(file.read())

        model_providers = {
            "OpenAI O4 Mini": "",
            "GPT-4.1": "",
            "ChatGPT-4o": "",
            "GPT-4.1 Mini": "",
            "GPT-4.1 Nano": "",
            "OpenAI O1": "",
            "Claude 3.7 Sonnet": "",
            "Claude 3.5 Haiku": "",
            "Gemini 2.5 Pro Preview": "",
            "Gemini 2.0 Flash": "",
            "Gemini 2.0 Flash Lite": "",
            "Gemini 2.5 Flash Preview": "",
            "Cohere Aya Vision 8B": "",
            "Cohere Aya Vision 32B": "",
            "Qwen 2.5 VL 7B": "",
            "Mistral Small 3.1 24b": "",
            "Llama 4 Scout 17B": "",
            "Llama 3 11B Vision": "",
            "Gemma 3 27b": "",
            "Claude 4 Sonnet": "",
            "Claude 4 Opus": "",
            "Mistral Medium 3": "",
            "Gemma 3 1B": "",
            "Mistral Small 3.1 24B": "",
            "Gemma 3 4B": "",
            "Phi 4 Multimodal": "",
            "Gemini 1.5 Flash": "",
            "Gemini 1.5 Pro": "",
            "Arcee.ai Spotlight": "",
        }
        # load from saved_results
        assessments_by_model = final_results["assessments_by_model"]
        model_results = final_results["model_results"]

        assessments = final_results["assessments"]

        assessment_categories = list(set([i["category"] for i in assessments]))
        assessment_categories.sort()
        for model_name, results in assessments_by_model.items():
            for assessment in results.values():
                times_by_model[model_name].append(float(assessment["time_taken"].replace("s", "")))
    else:
        model_providers = {
            "Claude 4 Sonnet": AnthropicModel(model_id="claude-sonnet-4-20250514"),
            "Claude 4 Opus": AnthropicModel(model_id="claude-opus-4-20250514"),
            "OpenAI O4 Mini": OpenAIModel(model_id="o4-mini"),
            "OpenAI O3": OpenAIModel(model_id="o3"),
            "GPT-4.1": OpenAIModel(model_id="gpt-4.1"),
            "ChatGPT-4o": OpenAIModel(model_id="chatgpt-4o-latest"),
            # "OpenAI O3": OpenAIModel(model_id="o3"),
            "GPT-4.1 Mini": OpenAIModel(model_id="gpt-4.1-mini"),
            "GPT-4.1 Nano": OpenAIModel(model_id="gpt-4.1-nano"),
            "OpenAI O1": OpenAIModel(model_id="o1"),
            "Llama 3 11B Vision": CustomOpenAIModel(model_id="meta-llama/Llama-3.2-11B-Vision-Instruct", base_url="https://router.huggingface.co/hf-inference/models/meta-llama/Llama-3.2-11B-Vision-Instruct/v1", api_key=os.environ.get("HUGGINGFACE_API_KEY")),
            "Gemma 3 27b": CustomOpenAIModel(
                model_id="google/gemma-3-27b-it-fast",
                base_url="https://router.huggingface.co/nebius/v1",
                api_key=os.environ.get("HUGGINGFACE_API_KEY"),
            ),
            "Llama 4 Scout 17B": CustomOpenAIModel(
                model_id="meta-llama/Llama-4-Scout-17B-16E-Instruct",
                base_url="https://router.huggingface.co/together/v1",
                api_key=os.environ.get("HUGGINGFACE_API_KEY"),
            ),
            "Claude 3.7 Sonnet": AnthropicModel(model_id="claude-3-7-sonnet-20250219"),
            "Claude 3.5 Haiku": AnthropicModel(model_id="claude-3-5-haiku-20241022"),
            "Gemini 2.5 Pro Preview": GeminiModel(model_id="gemini-2.5-pro-preview-03-25"),
            "Gemini 1.5 Flash": GeminiModel(model_id="gemini-1.5-flash"),
            "Gemini 1.5 Pro": GeminiModel(model_id="gemini-1.5-pro"),
            "Gemini 2.0 Flash": GeminiModel(model_id="gemini-2.0-flash"),
            "Gemini 2.0 Flash Lite": GeminiModel(model_id="gemini-2.0-flash-lite"),
            "Gemini 2.5 Flash Preview": GeminiModel(model_id="gemini-2.5-flash-preview-04-17"),
            "Cohere Aya Vision 8B": CohereModel(model_id="c4ai-aya-vision-8b"),
            "Cohere Aya Vision 32B": CohereModel(model_id="c4ai-aya-vision-32b"),
            "Qwen 2.5 VL 7B": CustomOpenAIModel(
                model_id="Qwen/Qwen2.5-VL-7B-Instruct",
                base_url="https://router.huggingface.co/hyperbolic/v1",
                api_key=os.environ.get("HUGGINGFACE_API_KEY"),
            ),
            "Mistral Medium 3": CustomOpenAIModel(
                model_id="mistralai/mistral-medium-3",
                base_url="https://openrouter.ai/api/v1",
                api_key=os.environ.get("OPENROUTER_API_KEY"),
            ),
            "Mistral Small 3.1 24B": CustomOpenAIModel(
                model_id="mistralai/mistral-small-3.1-24b-instruct",
                base_url="https://openrouter.ai/api/v1",
                api_key=os.environ.get("OPENROUTER_API_KEY"),
            ),
            "Llama 4 Maverick 17B": CustomOpenAIModel(
                model_id="meta-llama/llama-4-maverick",
                base_url="https://openrouter.ai/api/v1",
                api_key=os.environ.get("OPENROUTER_API_KEY"),
            ),
            "Gemma 3 4B": CustomOpenAIModel(
                model_id="google/gemma-3-4b-it",
                base_url="https://openrouter.ai/api/v1",
                api_key=os.environ.get("OPENROUTER_API_KEY"),
            ),
            "Phi 4 Multimodal": CustomOpenAIModel(
                model_id="microsoft/phi-4-multimodal-instruct",
                base_url="https://openrouter.ai/api/v1",
                api_key=os.environ.get("OPENROUTER_API_KEY"),
            ),
            "Arcee.ai Spotlight": CustomOpenAIModel(
                model_id="arcee-ai/spotlight",
                base_url="https://openrouter.ai/api/v1",
                api_key=os.environ.get("OPENROUTER_API_KEY"),
            ),
        }

        if is_in_incremental_mode:
            with open("./model_results.json", "r") as file:
                final_results = orjson.loads(file.read())

            calculated_models = set(final_results["model_results"].keys())
            # load from saved_results
            assessments_by_model = final_results["assessments_by_model"]
            model_results = final_results["model_results"]

            # assessments = final_results["assessments"]

            assessment_categories = list(set([i["category"] for i in assessments]))
            assessment_categories.sort()
            for model_name, results in assessments_by_model.items():
                for assessment in results.values():
                    times_by_model[model_name].append(float(assessment["time_taken"].replace("s", "")))

        models_to_run = [(model_name, model_class) for model_name, model_class in model_providers.items()]

        if is_in_incremental_mode:
            # filter models to run based on the ones that have not been calculated yet
            models_to_run = [
                (model_name, model_class)
                for model_name, model_class in models_to_run
                if model_name not in calculated_models
            ]
            images_to_run_by_model = {
                model_name: set(
                    assessment["file_name"]
                    for assessment in assessments
                    if assessment["file_name"] not in assessments_by_model.get(model_name, {})
                )
                for model_name, _ in model_providers.items()
            }
        else:
            images_to_run_by_model = {
                model_name: set(
                    assessment["file_name"]
                    for assessment in assessments
                )
                for model_name, _ in models_to_run
            }

        # add to models to run where file name > 0
        models_to_run = [
            (model_name, model_class)
            for model_name, model_class in model_providers.items()
            if len(images_to_run_by_model[model_name]) > 0
        ]
        print(models_to_run, "models to run")

        # print(images_to_run_by_model["Claude 4 Opus"])

        # print(models_to_run, "d")
        # print(images_to_run_by_model["Gemini 1.5 Pro"])
        # exit()§

        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = [
                executor.submit(run_model_with_prompt, model_name, model_class, assessment)
                for model_name, model_class in models_to_run
                for assessment in images_to_run_by_model[model_name]
            ]

            total_assessments = len(futures)
            print(models_to_run)
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=total_assessments,
                desc="Running assessments",
            ):
                model_name, assessment, result, answer, time_taken = future.result()
                print(model_name, assessment["file_name"], result, answer, time_taken)
                if result is None:
                    # print(
                    #     f"Skipping {model_name} for {assessment['file_name']} as no result is returned"
                    # )
                    continue
                times_by_model[model_name].append(time_taken)

                payload = {
                    "result": result,
                    "answer": answer,
                    "file_name": assessment["file_name"],
                    "time_taken": f"{time_taken:.2f}s",
                    **assessment,
                }

                payload["correct"] = normalise_output(result) == normalise_output(answer) or (
                    len(normalise_output(result)) > 1
                    and normalise_output(answer) in normalise_output(result)
                )
                if not assessments_by_model.get(model_name):
                    assessments_by_model[model_name] = {}

                assessments_by_model[model_name][assessment["file_name"]] = payload
        
        model_results = {}

    for model_name, results in assessments_by_model.items():
        total = len(results)
        correct = sum(1 for assessment in results.values() if assessment["correct"])

        model_results[model_name] = {
            "total": total,
            "correct": correct,
            "percentage": round(correct / total * 100, 1),
            "logo": logos[model_name],
            "average_time": f"{sum(times_by_model[model_name]) / len(times_by_model[model_name]):.2f}s",
        }

    # order model results by percentage
    model_results = dict(
        sorted(model_results.items(), key=lambda item: item[1]["percentage"], reverse=True)
    )

    average_times_by_model = {
        model_name: f"{sum(times) / len(times):.2f}s"
        for model_name, times in times_by_model.items()
    }

    assessment_categories = list(set([i["category"] for i in assessments]))
    assessment_categories.sort()

    assessments_by_model_by_category = defaultdict(lambda: defaultdict(list))
    result_assessments_by_model_by_category = defaultdict(lambda: defaultdict(dict))

    for model_name, results in assessments_by_model.items():
        for assessment in assessments:
            assessment_item = results.get(assessment["file_name"], {})
            if not assessment_item:
                continue
            assessments_by_model_by_category[model_name][assessment["category"]].append(
                {
                    "result": assessment_item["result"],
                    "correct": assessment_item["correct"],
                    "average_time": assessment_item["time_taken"],
                    **assessment,
                }
            )

    for model_name, categories in assessments_by_model_by_category.items():
        for category, assess_list in categories.items():
            result_assessments_by_model_by_category[model_name][category] = {
                "assessments": assess_list,
                "passed": sum(1 for assessment in assess_list if assessment["correct"]),
                "failed": sum(1 for assessment in assess_list if not assessment["correct"]),
                "total": len(assess_list),
                "passed_percentage": round(
                    sum(1 for assessment in assess_list if assessment["correct"])
                    / len(assess_list)
                    * 100,
                    1,
                ),
            }

    # sort result_assessments_by_model_by_category by passed #
    for model_name, categories in result_assessments_by_model_by_category.items():
        result_assessments_by_model_by_category[model_name] = dict(
            sorted(
                categories.items(),
                key=lambda item: item[1]["passed_percentage"],
                reverse=True,
            )
        )

    # turn into list
    model_results_list = [
        {
            "model_name": model_name,
            "total": results["total"],
            "correct": results["correct"],
            "percentage": results["percentage"],
            "logo": results["logo"],
            "average_time": results["average_time"],
        }
        for model_name, results in model_results.items()
    ]

    # set "postiion"
    for i, result in enumerate(model_results_list):
        if i == 0:
            result["position"] = 1
        elif i > 0 and result["percentage"] == model_results_list[i - 1]["percentage"]:
            result["position"] = model_results_list[i - 1]["position"]
        else:
            result["position"] = model_results_list[i - 1]["position"] + 1

    output = template.render(
        assessments_by_model=assessments_by_model,
        model_providers=model_providers,
        model_results=model_results_list,
        assessments=assessments,
        assessment_count=len(assessments),
        tasks=assessment_categories,
        task="all",
        title="Vision AI Checkup",
        open_or_closed_source=open_or_closed_source,
    )

    models = list(model_providers.keys())
    model_combinations = list(combinations(models, 2))

    # create page for each category, as task-name.html
    final_results = {"category_results": {}, "model_results": {}}
    final_results["category_results"]["all"] = model_results

    for category in assessment_categories:
        category_assessments = [
            assessment for assessment in assessments if assessment["category"] == category
        ]
        filtered_assessments_by_model = {
            model_name: {
                file_name: assessment
                for file_name, assessment in results.items()
                if assessment["category"] == category
            }
            for model_name, results in assessments_by_model.items()
        }
        category_model_results = {
            model_name: {
                "total": len(results),
                "correct": sum(1 for result in results.values() if result["correct"]),
                "percentage": round(
                    sum(1 for result in results.values() if result["correct"])
                    / (len(results) or 1)
                    * 100,
                    1,
                ),
                "logo": logos[model_name],
                "average_time": average_times_by_model[model_name],
            }
            for model_name, results in filtered_assessments_by_model.items()
        }
        category_model_results = dict(
            sorted(
                category_model_results.items(),
                key=lambda item: item[1]["percentage"],
                reverse=True,
            )
        )
        final_results["category_results"][category] = category_model_results

        # turn into list
        category_model_results_list = [
            {
                "model_name": model_name,
                "total": results["total"],
                "correct": results["correct"],
                "percentage": results["percentage"],
                "logo": results["logo"],
                "average_time": results["average_time"],
            }
            for model_name, results in category_model_results.items()
        ]

        # set "postiion"
        for i, result in enumerate(category_model_results_list):
            if i == 0:
                result["position"] = 1
            elif i > 0 and result["percentage"] == category_model_results_list[i - 1]["percentage"]:
                result["position"] = category_model_results_list[i - 1]["position"]
            else:
                result["position"] = category_model_results_list[i - 1]["position"] + 1

        category_output = template.render(
            assessments_by_model=assessments_by_model,
            model_providers=model_providers,
            model_results=category_model_results_list,
            assessments=category_assessments,
            assessment_count=len(category_assessments),
            tasks=assessment_categories,
            category=category,
            task=category.replace(" ", "-").lower(),
            title=f"Best {category} Models - Vision AI Checkup",
            description=f"Explore the best models for {category} tasks.",
        )

        with open(os.path.join(OUTPUT_DIR, f"{slugify(category)}.html"), "w") as file:
            file.write(category_output)


    with open(os.path.join(OUTPUT_DIR, "index.html"), "w") as file:
        file.write(output)

    for model_name, results in assessments_by_model.items():
        os.makedirs(os.path.join(OUTPUT_DIR, slugify(model_name)), exist_ok=True)

        with open(
            os.path.join(OUTPUT_DIR, f"{slugify(model_name)}/index.html"), "w"
        ) as file:
            results = sorted(
                results.values(),
                key=lambda x: (not x["correct"], x["assessment_name"], x["file_name"]),
            )

            model_results_json = {
                "by_category_results": result_assessments_by_model_by_category[model_name],
                "results": results
            }

            final_results["model_results"][model_name] = model_results_json

            # calculate what model is best at out of all models
            best_categories = []
            max_percentage = 0
            for category, category_results in result_assessments_by_model_by_category[model_name].items():
                if category_results["passed_percentage"] > max_percentage:
                    max_percentage = category_results["passed_percentage"]

            for category, category_results in result_assessments_by_model_by_category[model_name].items():
                if category_results["passed_percentage"] == max_percentage:
                    best_categories.append(category)

            file.write(
                card_template.render(
                    open_or_closed_source=open_or_closed_source,
                    model_name=model_name,
                    grid=True,
                    comparisons=[{"slug": f"/compare/{slugify(m1)}-vs-{slugify(m2)}/", "model_name": m2 if m1 == model_name else m1} for m1, m2 in model_combinations if m1 == model_name or m2 == model_name],
                    all_models=list(model_providers.keys()),
                    best_categories=best_categories,
                    results_csv_file=os.path.join(
                        OUTPUT_DIR, f"{slugify(model_name)}/results.csv"
                    ),
                    assessments=assessments,
                    results=results,
                    passed_percentage=round(
                        sum(1 for result in results if result["correct"])
                        / len(results)
                        * 100,
                        2,
                    ),
                    passed=sum(1 for result in results if result["correct"]),
                    failed=sum(1 for result in results if not result["correct"]),
                    total=len(results),
                    logo=logos[model_name],
                    by_category_results=result_assessments_by_model_by_category[model_name],
                    average_time=average_times_by_model[model_name],
                    title=f"{model_name} Results - Vision AI Checkup",
                    description=f"Explore the results of {model_name} on various vision tasks, from object understanding to document question answering.",
                    og_image="https://v1.screenshot.11ty.dev/" + quote_plus("https://visioncheckup.com/" + slugify(model_name)),
                )
            )

    saved_results = {
        "assessments_by_model": assessments_by_model,
        "model_results": model_results,
        "assessments": assessments,
        "assessment_count": len(assessments),
        "tasks": assessment_categories,
        "final_results": final_results,
    }
    # TypeError: Object of type bytes is not JSON serializable
    # delete bytes recursively
    def delete_bytes(obj):
        if isinstance(obj, bytes):
            try:
                return obj.decode("utf-8")
            except UnicodeDecodeError:
                return ""
        elif isinstance(obj, dict):
            return {key: delete_bytes(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [delete_bytes(item) for item in obj]
        else:
            return obj
        
    saved_results = delete_bytes(saved_results)

    with open("model_results.json", "w") as file:
        file.write(json.dumps(saved_results, indent=4))

    for assessment in assessments:
        src = os.path.join(BASE_IMAGE_DIR, assessment["file_name"])
        dst = os.path.join(OUTPUT_DIR, "images", assessment["file_name"])
        if os.path.exists(src):
            shutil.copy(src, dst)
        # copy compressed/+ filename
        compressed_src = os.path.join(BASE_IMAGE_DIR, "compressed/", assessment["file_name"].replace(".png", ".jpeg"))
        # print(compressed_src)
        compressed_dst = os.path.join(OUTPUT_DIR, "images", "compressed/", assessment["file_name"].replace(".png", ".jpeg"))
        if os.path.exists(compressed_src):
            # print(f"Copying {compressed_src} to {compressed_dst}")
            os.makedirs(os.path.join(OUTPUT_DIR, "images", "compressed/"), exist_ok=True)
            shutil.copy(compressed_src, compressed_dst)

    prompts_output = prompts_template.render(
        assessments=assessments,
        assessment_count=len(assessments),
        tasks=assessment_categories,
        full_width=True,
        task_counts={
            category: sum(1 for assessment in assessments if assessment["category"] == category)
            for category in assessment_categories
        },
        title="Prompts | Vision AI Checkup",
        description="Explore prompts used to evaluate various vision models on different tasks.",
        og_image="https://visioncheckup.com/prompts",
    )

    with open(os.path.join(OUTPUT_DIR, "prompts/index.html"), "w") as file:
        file.write(prompts_output)

    # create pages for each assessment
    for assessment in assessments:
        model_results = []
        for model_name, results in assessments_by_model.items():
            if assessment["file_name"] in results:
                result = results[assessment["file_name"]]
                # print(
                #     f"Creating page for {assessment['file_name']} with {model_name} - {result['correct']}"
                # )
                model_results.append(
                    {
                        "model_name": model_name,
                        "result": result["result"],
                        "answer": result["answer"],
                        "correct": result["correct"],
                        "time_taken": result["time_taken"],
                        # "bbox_image": result.get("bbox_image"),
                    }
                )

        model_results = sorted(
            model_results,
            key=lambda x: (not x["correct"], x["model_name"]),
        )

        assessment_output = assessment_template.render(
            assessment=assessment,
            model_results=model_results,
            grid=True,
            correct= all(
                result["correct"] for result in model_results
            ),  # check if all models passed
            passed_count=sum(1 for result in model_results if result["correct"]),
            failed_count=sum(1 for result in model_results if not result["correct"]),
            total_count=len(model_results),
            title=f"{assessment['assessment_name']} - Vision AI Checkup",
            description=f"View the results of {assessment['assessment_name']} when run against various SOTA vision models.",
            og_image="https://v1.screenshot.11ty.dev/" + quote_plus("https://visioncheckup.com/assessments/" + slugify(assessment["assessment_name"]))
        )
        os.makedirs(
            os.path.join(OUTPUT_DIR, "assessments", slugify(assessment["assessment_name"])),
            exist_ok=True,
        )

        with open(
            os.path.join(
                OUTPUT_DIR, "assessments", f"{slugify(assessment['assessment_name'])}/index.html"
            ),
            "w",
        ) as file:
            file.write(assessment_output)

    for model1, model2 in model_combinations:
        # print(f"Comparing {model1} and {model2}")
        by_category_results = defaultdict(lambda: defaultdict(dict))

        for category in assessment_categories:
            model1_results = []
            model2_results = []
            for assessment in assessments:
                if assessment["category"] != category:
                    continue
                # skip if not assessments by model
                if not assessments_by_model.get(model1) or not assessments_by_model.get(model2):
                    continue
                model1_result = assessments_by_model[model1].get(assessment["file_name"])
                model2_result = assessments_by_model[model2].get(assessment["file_name"])

                if model1_result:
                    model1_results.append(model1_result)
                if model2_result:
                    model2_results.append(model2_result)

            by_category_results[category]["model1"] = {
                "assessments": model1_results,
                "model_name": model1,
                "passed": sum(1 for result in model1_results if result["correct"]),
                "failed": sum(1 for result in model1_results if not result["correct"]),
                "total": len(model1_results),
                "passed_percentage": round(
                    sum(1 for result in model1_results if result["correct"])
                    / (len(model1_results) or 1)
                    * 100,
                    1,
                ),
                "avg_time": f"{sum(float(result["time_taken"].replace("s", "")) for result in model1_results) / (len(model1_results) or 1):.2f}s"
            }

            by_category_results[category]["model2"] = {
                "assessments": model2_results,
                "model_name": model2,
                "passed": sum(1 for result in model2_results if result["correct"]),
                "failed": sum(1 for result in model2_results if not result["correct"]),
                "total": len(model2_results),
                "passed_percentage": round(
                    sum(1 for result in model2_results if result["correct"])
                    / (len(model2_results) or 1)
                    * 100,
                    1,
                ),
                "avg_time": f"{sum(float(result["time_taken"].replace("s", "")) for result in model2_results) / (len(model2_results) or 1):.2f}s",
            }

        # create a compare page for each model combination
        # render the compare template with the model data and results
        # if not times, skip
        if not times_by_model.get(model1) or not times_by_model.get(model2):
            # print(f"Skipping comparison for {model1} and {model2} as no results found")
            continue
        compare_output = compare_template.render(
            model1=model1,
            model2=model2,
            avg_time_model1=f"{sum(times_by_model[model1]) / len(times_by_model[model1]):.2f}s",
            avg_time_model2=f"{sum(times_by_model[model2]) / len(times_by_model[model2]):.2f}s",
            passed_percentage_model1=round(
                sum(1 for result in assessments_by_model[model1].values() if result["correct"])
                / (len(assessments_by_model[model1]) or 1)
                * 100,
                1,
            ),
            passed_percentage_model2=round(
                sum(1 for result in assessments_by_model[model2].values() if result["correct"])
                / (len(assessments_by_model[model2]) or 1)
                * 100,
                1,
            ),
            passed_count_model1=sum(
                1 for result in assessments_by_model[model1].values() if result["correct"]
            ),
            passed_count_model2=sum(
                1 for result in assessments_by_model[model2].values() if result["correct"]
            ),
            total_model1=len(assessments_by_model[model1]),
            total_model2=len(assessments_by_model[model2]),
            model1_results=assessments_by_model[model1],
            model2_results=assessments_by_model[model2],
            by_category_results=by_category_results,
            assessments=assessments,
            title=f"{model1} vs {model2} - Vision AI Checkup",
            description=f"See how {model1} and {model2} compare on defect detection, document understanding, VQA, and more.",
            og_image="https://v1.screenshot.11ty.dev/" + quote_plus(
                f"https://visioncheckup.com/compare/{slugify(model1)}-vs-{slugify(model2)}/"
            ),
        )

        os.makedirs(os.path.join(OUTPUT_DIR, "compare"), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, "compare", f"{slugify(model1)}-vs-{slugify(model2)}"), exist_ok=True)
        with open(os.path.join(OUTPUT_DIR, "compare", f"{slugify(model1)}-vs-{slugify(model2)}/index.html"), "w") as file:
            file.write(compare_output)

    urls = []

    urls.append("https://visioncheckup.com/")

    for assessment in assessments:
        urls.append(f"https://visioncheckup.com/assessments/{slugify(assessment['assessment_name'])}/")

    for model_name in model_providers.keys():
        urls.append(f"https://visioncheckup.com/{slugify(model_name)}/")

    # add compare pages urls
    for model1, model2 in model_combinations:
        urls.append(f"https://visioncheckup.com/compare/{slugify(model1)}-vs-{slugify(model2)}/")

    urls.append("https://visioncheckup.com/prompts/")

    # generate sitemap.xml
    sitemap_output = sitemap_template.render(
        site_url="https://visioncheckup.com",
        urls=urls,
        build_date=time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
    )
    with open(os.path.join(OUTPUT_DIR, "sitemap.xml"), "w") as file:
        file.write(sitemap_output)

    assets_dir = "assets/"
    if os.path.exists(assets_dir):
        shutil.copytree(assets_dir, OUTPUT_DIR, dirs_exist_ok=True)

class TemplateChangeHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if not event.is_directory:
            print(f"[Modified] {event.src_path} – running main()")
            main()

    def on_created(self, event):
        if not event.is_directory:
            print(f"[Created]  {event.src_path} – running main()")
            main()

if __name__ == "__main__":
    main()
    # if --watch flag is set, watch for changes in the templates directory
    if "--watch" in os.sys.argv:
        event_handler = TemplateChangeHandler()
        observer = Observer()
        observer.schedule(event_handler, path="templates/", recursive=True)
        print("Watching for changes in templates directory...")
        observer.start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            observer.stop()
        print("Stopping observer...")
        observer.join()