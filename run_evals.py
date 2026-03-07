import concurrent.futures
import glob
import json
import os
import string
import time
from collections import defaultdict

from tqdm import tqdm

from model_registry import MODEL_REGISTRY


BASE_IMAGE_DIR = "images/"


def normalise_output(output):
    if not output:
        return ""

    output = str(output).strip().lower()
    output = output.translate(str.maketrans("", "", string.punctuation))

    return output.strip().replace(" ", "")


def compare_json_values(val1, val2):
    """
    Recursively compare two values.
    - If both are dicts, compare keys and values.
    - If both are lists, compare elements (assuming order matters for now).
    - If strings, use normalise_output.
    - Otherwise, use equality.
    """
    if isinstance(val1, dict) and isinstance(val2, dict):
        if set(val1.keys()) != set(val2.keys()):
            return False
        for k in val1:
            if not compare_json_values(val1[k], val2[k]):
                return False
        return True

    elif isinstance(val1, list) and isinstance(val2, list):
        if len(val1) != len(val2):
            return False
        for v1, v2 in zip(val1, val2):
            if not compare_json_values(v1, v2):
                return False
        return True

    elif isinstance(val1, str) or isinstance(val2, str):
        return normalise_output(val1) == normalise_output(val2)

    else:
        return val1 == val2


def compare_outputs(predicted_str, ground_truth_str):
    # Check if ground_truth looks like JSON
    gt_is_json = False
    if isinstance(ground_truth_str, str):
        clean_gt = ground_truth_str.strip()
        # Replace curly quotes with straight quotes for JSON parsing
        clean_gt = clean_gt.replace("\u201c", '"').replace("\u201d", '"')
        if clean_gt.startswith("{") and clean_gt.endswith("}"):
            gt_is_json = True
            ground_truth_str = clean_gt

    if gt_is_json:
        try:
            gt_json = json.loads(ground_truth_str)

            try:
                pred_json = json.loads(predicted_str)
            except:
                return normalise_output(predicted_str) == normalise_output(ground_truth_str)

            return compare_json_values(pred_json, gt_json)

        except Exception:
            pass

    return normalise_output(predicted_str) == normalise_output(ground_truth_str)


def _run_model_with_prompt(model_name, model, assessment, assessments):
    if isinstance(assessment, str):
        assessment = [a for a in assessments if a["file_name"] == assessment][0]

    with open(
        os.path.join(BASE_IMAGE_DIR, assessment["file_name"]), "rb"
    ) as image_file:
        assessment["image"] = image_file.read()
    start_time = time.time()

    result = model.run_with_retry(
        assessment["image"],
        assessment["prompt"] + '\nReturn the result in JSON format, e.g. {"answer": "your answer"}.',
        image_name=os.path.join(BASE_IMAGE_DIR, assessment["file_name"]),
        structured_output_format={"type": "json_object"}
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
            assessment["prompt"] + '\nReturn the result in JSON format, e.g. {"answer": "your answer"}.',
            image_name=os.path.join(BASE_IMAGE_DIR, "compressed/", assessment["file_name"].replace(".png", ".jpeg")),
            structured_output_format={"type": "json_object"}
        )

    end_time = time.time()
    assessment["image"] = None

    answer = assessment["answer"]

    time_taken = end_time - start_time

    return model_name, assessment, result, answer, time_taken


def run_evals(assessments, options):
    """Run model evaluations and return results.

    Returns (assessments_by_model, times_by_model, model_providers, added_on)
    """
    is_in_incremental_mode = options.incremental

    assessments_by_model = defaultdict(lambda: defaultdict(list))
    times_by_model = defaultdict(list)
    added_on = {}

    # if data/results directory exists, load results from there
    if os.path.exists("data/results") and not is_in_incremental_mode:
        for file_path in glob.glob("data/results/result_*.json"):
            with open(file_path, "r") as f:
                data = json.load(f)
                if "model_name" in data:
                    m_name = data["model_name"]
                    assessments_by_model[m_name] = data["assessments"]
                else:
                    pass

        # Load metadata
        if os.path.exists("data/metadata.json"):
            with open("data/metadata.json", "r") as f:
                metadata = json.load(f)
                added_on = metadata.get("added_on", {})

        model_providers = {name: "" for name in MODEL_REGISTRY}

        for model_name, results in assessments_by_model.items():
            for assessment in results.values():
                times_by_model[model_name].append(float(assessment["time_taken"].replace("s", "")))
    else:
        model_providers = {
            name: cfg["provider"]()
            for name, cfg in MODEL_REGISTRY.items()
            if cfg.get("provider")
        }

        if is_in_incremental_mode:
            # Load results from data/results
            if os.path.exists("data/results"):
                for file_path in glob.glob("data/results/result_*.json"):
                    with open(file_path, "r") as f:
                        data = json.load(f)
                        if "model_name" in data:
                            m_name = data["model_name"]
                            assessments_by_model[m_name] = data["assessments"]

             # Load metadata
            if os.path.exists("data/metadata.json"):
                with open("data/metadata.json", "r") as f:
                    metadata = json.load(f)
                    added_on = metadata.get("added_on", {})

            calculated_models = set(assessments_by_model.keys())

            for model_name, results in assessments_by_model.items():
                for assessment in results.values():
                    times_by_model[model_name].append(float(assessment["time_taken"].replace("s", "")))

        models_to_run = [(model_name, model_class) for model_name, model_class in model_providers.items()]

        if options.model:
            import re
            models_to_run = [
                (model_name, model_class)
                for model_name, model_class in models_to_run
                if re.search(options.model, model_name, re.IGNORECASE)
            ]

        if options.build_only:
            models_to_run = []

        if is_in_incremental_mode:
            # filter models to run based on the ones that have not been calculated yet
            models_to_run = [
                (model_name, model_class)
                for model_name, model_class in models_to_run
                if model_name not in calculated_models
            ]

            print(f"Models to run: {len(models_to_run)}")
            print("Models to run: ", models_to_run)

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

        with concurrent.futures.ThreadPoolExecutor(max_workers=options.concurrency) as executor:
            futures = [
                executor.submit(_run_model_with_prompt, model_name, model_class, assessment, assessments)
                for model_name, model_class in models_to_run
                for assessment in images_to_run_by_model[model_name]
            ]

            total_assessments = len(futures)

            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=total_assessments,
                desc="Running assessments",
            ):
                model_name, assessment, result, answer, time_taken = future.result()
                if result is None:
                    continue
                times_by_model[model_name].append(time_taken)

                try:
                    print('result', result)
                    # try to parse json
                    if isinstance(result, dict):
                        val = result.get("answer", result)
                        if isinstance(val, (dict, list)):
                            parsed_answer = json.dumps(val)
                        else:
                            parsed_answer = str(val)
                    elif isinstance(result, str):
                        # clean up markdown code blocks if present
                        if "```json" in result:
                            result = result.split("```json")[1].split("```")[0].strip()
                        elif "```" in result:
                            result = result.split("```")[1].split("```")[0].strip()

                        parsed = json.loads(result)
                        val = parsed.get("answer", parsed)
                        if isinstance(val, (dict, list)):
                            parsed_answer = json.dumps(val)
                        else:
                            parsed_answer = str(val)
                    else:
                        parsed_answer = str(result)
                except Exception as e:
                    print(f"Failed to parse result for {model_name} on {assessment['file_name']}: {e}")
                    print(f"Raw Result: {result}")
                    parsed_answer = str(result)

                payload = {
                    "result": result if isinstance(result, (dict, list)) else result,
                    "answer": answer,
                    "parsed_answer": parsed_answer,
                    "file_name": assessment["file_name"],
                    "time_taken": f"{time_taken:.2f}s",
                    **assessment,
                }

                # Strict checking
                payload["correct"] = compare_outputs(parsed_answer, answer)
                if not assessments_by_model.get(model_name):
                    assessments_by_model[model_name] = {}

                assessments_by_model[model_name][assessment["file_name"]] = payload

    return assessments_by_model, times_by_model, model_providers, added_on
