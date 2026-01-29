import json
import os
from io import BytesIO

from google import genai
from google.genai import types
from PIL import Image

from .model import Model

GEMINI_TEMPERATURE = 0.1


def parse_json(json_output: str):
    # Parsing out the markdown fencing
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line == "```json":
            json_output = "\n".join(
                lines[i + 1 :]
            )  # Remove everything before "```json"
            json_output = json_output.split("```")[
                0
            ]  # Remove everything after the closing "```"
            break  # Exit the loop once "```json" is found
    return json_output


class GeminiModel(Model):
    def __init__(self, model_id: str, enable_code_execution: bool = False):
        self.model_id = model_id
        self.enable_code_execution = enable_code_execution
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    def run(
        self,
        image: str,
        prompt: str,
        image_name="image.png",
        structured_output_format=None,
    ):
        result = self.client.models.generate_content(
            model=self.model_id,
            config=types.GenerateContentConfig(
                temperature=GEMINI_TEMPERATURE,
                tools=[types.Tool(code_execution=types.ToolCodeExecution)] if self.enable_code_execution else None,
                response_mime_type="application/json" if structured_output_format else "text/plain",
            ),
            contents=[
                prompt,
                types.Part.from_bytes(
                    data=image,
                    mime_type="image/"
                    + image_name.split(".")[-1].replace("jpg", "jpeg"),
                ),
            ],
        )

        if "xyxy" in prompt:
            # If the prompt contains "xyxy", we assume it's a request for structured output
            bounding_boxes = parse_json(result.text)
            final_bboxes = []

            with BytesIO(image) as img_io:
                img = Image.open(img_io)
                width, height = img.size

            for i, bounding_box in enumerate(json.loads(bounding_boxes)):
                abs_y1 = int(bounding_box["box_2d"][0] / 1000 * height)
                abs_x1 = int(bounding_box["box_2d"][1] / 1000 * width)
                abs_y2 = int(bounding_box["box_2d"][2] / 1000 * height)
                abs_x2 = int(bounding_box["box_2d"][3] / 1000 * width)

                if abs_x1 > abs_x2:
                    abs_x1, abs_x2 = abs_x2, abs_x1

                if abs_y1 > abs_y2:
                    abs_y1, abs_y2 = abs_y2, abs_y1

                final_bboxes.append([abs_x1, abs_y1, abs_x2, abs_y2])

            return final_bboxes

        return result.text
