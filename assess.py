import csv
import os
import shutil
import time
from dotenv import load_dotenv

load_dotenv()

import optparse
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from build_site import slugify
from run_evals import run_evals
from build_site import build_site

OUTPUT_DIR = "docs"


def parse_args():
    parser = optparse.OptionParser()
    parser.add_option(
        "--incremental",
        action="store_true",
        dest="incremental",
        default=False,
        help="Run in incremental mode, only updating changed files.",
    )
    parser.add_option(
        "--watch",
        action="store_true",
        dest="watch",
        default=False,
        help="Watch for changes in the directory and update the docs incrementally.",
    )
    parser.add_option(
        "--build-only",
        action="store_true",
        dest="build_only",
        default=False,
        help="Only build the documentation, skip inference.",
    )
    parser.add_option(
        "--model",
        dest="model",
        default=None,
        help="Run only the specified model (partial match allowed).",
    )
    parser.add_option(
        "--concurrency",
        dest="concurrency",
        type="int",
        default=1,
        help="Number of concurrent workers for inference.",
    )
    options, _ = parser.parse_args()
    return options


options = parse_args()

if options.incremental:
    print("Running in incremental mode. Only changed files will be updated.")

if options.build_only:
    print("Running in build-only mode. Skipping inference.")

if os.path.exists("docs"):
    shutil.rmtree("docs")


def main():
    # Load assessments from CSV
    with open("prompts.csv", "r") as file:
        reader = csv.DictReader(file)
        assessments = list(reader)
        for assessment in assessments:
            assessment["slug"] = slugify(assessment["assessment_name"])

    # Run evals (or load cached results)
    assessments_by_model, times_by_model, model_providers, added_on = run_evals(assessments, options)

    # Build the site
    build_site(assessments, assessments_by_model, times_by_model, model_providers, added_on)


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
    # if --watch flag is set, watch for changes in the templates directory and data/results
    if "--watch" in os.sys.argv:
        event_handler = TemplateChangeHandler()
        observer = Observer()
        observer.schedule(event_handler, path="templates/", recursive=True)
        if os.path.exists("data/results"):
            observer.schedule(event_handler, path="data/results/", recursive=False)
        print("Watching for changes in templates and data/results directories...")
        observer.start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            observer.stop()
        print("Stopping observer...")
        observer.join()
