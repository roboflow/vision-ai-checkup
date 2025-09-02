import base64

from openai import BadRequestError, OpenAI

from .model import Model

OPENAI_TEMPERATURE = 0.1
SKIP_TEMPERATURE = [
    "gpt-5-2025-08-07",
    "o4-mini",
    "chatgpt-4o-latest",
    "o3",
    "o1",
    "o3-pro",
    "o4-mini-high",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-5-chat",
    "gpt-5",
]

# Models that support OpenAI "reasoning_effort" parameter.
# Includes O-series and GPT-5 family.
ALLOWED_REASONING_MODELS = {
    "o3",
    "o4-mini",
    "o1",
    "o3-pro",
    "gpt-5-2025-08-07",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-5-chat",
    "gpt-5",
}


class OpenAIModel(Model):
    def __init__(self, model_id: str, reasoning_effort: str = "medium"):
        self.model_id = model_id
        self.client = OpenAI()
        self.reasoning_effort = reasoning_effort

    def run(
        self,
        image: str,
        prompt: str,
        structured_output_format=None,
        image_name: str = "image.png",
        reasoning_effort: str = "medium",
    ):
        image_dtype = "image/" + image_name.split(".")[-1].replace("jpg", "jpeg")
        if structured_output_format:
            try:
                kwargs = {
                    "model": self.model_id,
                    "messages": [
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
                        }
                    ],
                    "temperature": (
                        OPENAI_TEMPERATURE if self.model_id not in SKIP_TEMPERATURE else 1
                    ),
                    "response_format": structured_output_format,
                }
                if self.model_id in ALLOWED_REASONING_MODELS:
                    kwargs["reasoning_effort"] = reasoning_effort

                return (
                    self.client.beta.chat.completions.parse(**kwargs).choices[0].message.parsed
                    or {}
                )
            except BadRequestError as e:
                print(f"Error parsing structured output: {e}")
                pass

        # if reasoning_effort != "medium":
        kwargs = {
            "model": self.model_id,
            "messages": [
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
                }
            ],
            "temperature": (
                OPENAI_TEMPERATURE if self.model_id not in SKIP_TEMPERATURE else 1
            ),
        }
        if self.model_id in ALLOWED_REASONING_MODELS:
            kwargs["reasoning_effort"] = reasoning_effort

        completion = self.client.chat.completions.create(**kwargs)

        return completion.choices[0].message.content
        # else:
        #     completion = self.client.responses.create(
        #         model=self.model_id,
        #         input=[
        #             {
        #                 "role": "user",
        #                 "content": [
        #                     {"type": "input_text", "text": prompt},
        #                     {
        #                         "type": "input_image",
        #                         "image_url": f"data:{image_dtype};base64,{base64.b64encode(image).decode()}"
        #                     },
        #                 ],
        #             },
        #         ],
        #         temperature=(
        #             OPENAI_TEMPERATURE if self.model_id not in SKIP_TEMPERATURE else 1
        #         ),
        #     )
        #     return completion.output_text
        # # print(f"OpenAI completion: {completion}")
