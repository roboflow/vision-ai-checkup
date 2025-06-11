import base64

from openai import BadRequestError, OpenAI

from .model import Model

OPENAI_TEMPERATURE = 0.1
SKIP_TEMPERATURE = ["o4-mini", "chatgpt-4o-latest", "o3", "o1", "o3-pro"]


class OpenAIModel(Model):
    def __init__(self, model_id: str, ):
        self.model_id = model_id
        self.client = OpenAI()

    def run(
        self,
        image: str,
        prompt: str,
        structured_output_format=None,
        image_name: str = "image.png",
    ):
        image_dtype = "image/" + image_name.split(".")[-1].replace("jpg", "jpeg")
        if structured_output_format:
            try:
                return (
                    self.client.beta.chat.completions.parse(
                        model=self.model_id,
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt},
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:{image_dtype};base64,{base64.b64encode(image).decode()}"
                                        },
                                    },
                                ],
                            },
                        ],
                        temperature=(
                            OPENAI_TEMPERATURE
                            if self.model_id not in SKIP_TEMPERATURE
                            else 1
                        ),
                        response_format=structured_output_format,
                    )
                    .choices[0]
                    .message.parsed
                    or {}
                )
            except BadRequestError as e:
                print(f"Error parsing structured output: {e}")
                pass
        completion = self.client.responses.create(
            model=self.model_id,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {
                            "type": "input_image",
                            "image_url": f"data:{image_dtype};base64,{base64.b64encode(image).decode()}"
                        },
                    ],
                },
            ],
            temperature=(
                OPENAI_TEMPERATURE if self.model_id not in SKIP_TEMPERATURE else 1
            ),
        )

        return completion.output_text
