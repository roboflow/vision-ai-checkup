import csv
import datetime
import json
import os
import shutil
import time
from collections import defaultdict
from itertools import combinations
from urllib.parse import quote_plus

from jinja2 import Environment, FileSystemLoader

from model_registry import MODEL_REGISTRY


OUTPUT_DIR = "docs"
BASE_IMAGE_DIR = "images/"

try:
    with open("models.csv", "r") as file:
        reader = csv.DictReader(file)
        model_info = list(reader)
except FileNotFoundError:
    model_info = []

logos = {name: cfg["logo"] for name, cfg in MODEL_REGISTRY.items()}


def slugify(value):
    """Convert a string to a slug."""
    value = value.lower()
    value = value.replace(" ", "-")
    value = "".join(
        c if c.isalnum() or c == "-" else "-" for c in value
    )
    return value


def build_site(assessments, assessments_by_model, times_by_model, model_providers, added_on):
    """Build the docs/ site from eval results."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "prompts"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "assessments"), exist_ok=True)

    # Copy local icons
    if os.path.exists("assets/images/z-icon.svg"):
        shutil.copy("assets/images/z-icon.svg", os.path.join(OUTPUT_DIR, "images", "z-icon.svg"))
    if os.path.exists("assets/images/kimi-icon.ico"):
        shutil.copy("assets/images/kimi-icon.ico", os.path.join(OUTPUT_DIR, "images", "kimi-icon.ico"))
    if os.path.exists("assets/images/xai-icon.svg"):
        shutil.copy("assets/images/xai-icon.svg", os.path.join(OUTPUT_DIR, "images", "xai-icon.svg"))

    env = Environment(loader=FileSystemLoader("templates"))
    template = env.get_template("index.html")
    card_template = env.get_template("card.html")
    prompts_template = env.get_template("prompts.html")
    assessment_template = env.get_template("assessment.html")
    compare_template = env.get_template("compare.html")
    sitemap_template = env.get_template("sitemap.xml")
    llms_txt_template = env.get_template("llms.txt")

    # Compute model results
    model_results = {}
    for model_name, results in assessments_by_model.items():
        total = len(results)
        correct = sum(1 for assessment in results.values() if assessment["correct"])

        model_results[model_name] = {
            "total": total,
            "correct": correct,
            "percentage": round(correct / total * 100, 1),
            "logo": logos.get(model_name, ""),
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
            "is_open_source": MODEL_REGISTRY.get(model_name, {}).get("open_weights", False),
            "weights_size": MODEL_REGISTRY.get(model_name, {}).get("weights_size", ""),
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

    saved_results = model_results.copy()
    seven_days_ago = (datetime.date.today() - datetime.timedelta(days=7)).isoformat()
    new_models = [model_name for model_name, date in saved_results.get("added_on", {}).items() if date > seven_days_ago]

    output = template.render(
        assessments_by_model=assessments_by_model,
        model_providers=model_providers,
        model_results=model_results_list,
        assessments=assessments,
        new_models=new_models,
        model_dates=saved_results.get("added_on", {}),
        assessment_count=len(assessments),
        tasks=assessment_categories,
        task="all",
        title="Vision AI Checkup",
        added_models=[m for m in model_results.get("added_on", []) if m == datetime.date.today().isoformat()],
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
                "logo": logos.get(model_name, ""),
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
                "is_open_source": MODEL_REGISTRY.get(model_name, {}).get("open_weights", False),
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
            model_dates=saved_results.get("added_on", {}),
            new_models=new_models,
            model_results=category_model_results_list,
            assessments=category_assessments,
            assessment_count=len(category_assessments),
            tasks=assessment_categories,
            category=category,
            task=category.replace(" ", "-").lower(),
            title=f"Best {category} Models - Vision AI Checkup",
            description=f"Explore the best models for {category} tasks.",
            leaderboard_title=f"Best {category} VLMs"
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
                    model_description=([model_info_item["description"] for model_info_item in model_info if model_info_item["model_name"] == model_name] + [""])[0],
                    model_name=model_name,
                    playground_slug=MODEL_REGISTRY.get(model_name, {}).get("playground_slug", ""),
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
                    logo=logos.get(model_name, ""),
                    by_category_results=result_assessments_by_model_by_category[model_name],
                    average_time=average_times_by_model[model_name],
                    title=f"{model_name} Results - Vision AI Checkup",
                    description=f"Explore the results of {model_name} on various vision tasks, from object understanding to document question answering.",
                    og_image="https://v1.screenshot.11ty.dev/" + quote_plus("https://visioncheckup.com/" + slugify(model_name)),
                )
            )

    # Save metadata and results
    saved_results_data = {
        "assessments_by_model": assessments_by_model,
        "model_results": model_results,
        "assessments": assessments,
        "assessment_count": len(assessments),
        "tasks": assessment_categories,
        "final_results": final_results,
        "added_on": saved_results.get("added_on", {}) if saved_results else {},
    }

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

    saved_results_data = delete_bytes(saved_results_data)

    # Check added_on
    if not added_on:
         june_first = datetime.datetime(2025, 6, 1).isoformat()
         added_on_to_save = {model_name: june_first for model_name in assessments_by_model.keys()}
    else:
        added_on_to_save = added_on

    with open("data/metadata.json", "w") as file:
        json.dump({"added_on": added_on_to_save}, file, indent=4)

    # Save each model result
    for model_name, results in assessments_by_model.items():
        slug = slugify(model_name)
        file_path = f"data/results/result_{slug}.json"

        file_content = {
            "model_name": model_name,
            "assessments": results
        }
        with open(file_path, "w") as file:
            json.dump(file_content, file, indent=4)

    for assessment in assessments:
        src = os.path.join(BASE_IMAGE_DIR, assessment["file_name"])
        dst = os.path.join(OUTPUT_DIR, "images", assessment["file_name"])
        if os.path.exists(src):
            shutil.copy(src, dst)
        # copy compressed/+ filename
        compressed_src = os.path.join(BASE_IMAGE_DIR, "compressed/", assessment["file_name"].replace(".png", ".jpeg"))
        compressed_dst = os.path.join(OUTPUT_DIR, "images", "compressed/", assessment["file_name"].replace(".png", ".jpeg"))
        if os.path.exists(compressed_src):
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
        assessment_model_results = []
        for model_name, results in assessments_by_model.items():
            if assessment["file_name"] in results:
                result = results[assessment["file_name"]]
                assessment_model_results.append(
                    {
                        "model_name": model_name,
                        "result": result["result"],
                        "answer": result["answer"],
                        "correct": result["correct"],
                        "time_taken": result["time_taken"],
                    }
                )

        assessment_model_results = sorted(
            assessment_model_results,
            key=lambda x: (not x["correct"], x["model_name"]),
        )

        assessment_output = assessment_template.render(
            assessment=assessment,
            model_results=assessment_model_results,
            grid=True,
            correct=all(
                result["correct"] for result in assessment_model_results
            ),
            passed_count=sum(1 for result in assessment_model_results if result["correct"]),
            failed_count=sum(1 for result in assessment_model_results if not result["correct"]),
            total_count=len(assessment_model_results),
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
        by_category_results = defaultdict(lambda: defaultdict(dict))

        for category in assessment_categories:
            model1_results = []
            model2_results = []
            for assessment in assessments:
                if assessment["category"] != category:
                    continue
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
                "avg_time": f"{sum(float(result['time_taken'].replace('s', '')) for result in model1_results) / (len(model1_results) or 1):.2f}s"
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
                "avg_time": f"{sum(float(result['time_taken'].replace('s', '')) for result in model2_results) / (len(model2_results) or 1):.2f}s",
            }

        if not times_by_model.get(model1) or not times_by_model.get(model2):
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

    llms_txt = llms_txt_template.render(
        leaderboards={title: url for title, url in zip(assessment_categories, [f"/{slugify(category)}/" for category in assessment_categories])},
        models={title: url for title, url in zip(model_providers.keys(), [f"/{slugify(model_name)}/" for model_name in model_providers.keys()])},
        comparisons={title: url for title, url in zip([f"{m1} vs {m2}" for m1, m2 in model_combinations], [f"/compare/{slugify(m1)}-vs-{slugify(m2)}/" for m1, m2 in model_combinations])},
    )
    with open(os.path.join(OUTPUT_DIR, "llms.txt"), "w") as file:
        file.write(llms_txt)

    assets_dir = "assets/"
    if os.path.exists(assets_dir):
        shutil.copytree(assets_dir, OUTPUT_DIR, dirs_exist_ok=True)
