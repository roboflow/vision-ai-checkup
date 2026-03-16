import base64

from openai import BadRequestError, OpenAI

from .model import Model

OPENAI_TEMPERATURE = 0.1


class CustomOpenAIModel(Model):
    def __init__(self, model_id: str, base_url: str, api_key: str, supports_response_format: bool = True):
        self.model_id = model_id
        self.supports_response_format = supports_response_format
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )

    def run(
        self,
        image: str,
        prompt: str,
        structured_output_format=None,
        image_name: str = "image.png",
    ):
        image_dtype = "image/" + image_name.split(".")[-1].replace("jpg", "jpeg")

        kwargs = {}
        if self.supports_response_format and structured_output_format:
            kwargs["response_format"] = structured_output_format

        completion = self.client.chat.completions.create(
            model=self.model_id,
            **kwargs,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{image_dtype};base64,{base64.b64encode(image).decode()}"
                            },
                        },
                    ],
                }
            ],
            max_tokens=512,
        )

        return completion.choices[0].message.content
