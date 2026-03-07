import base64

import requests

from .model import Model


class SmolVLMModel(Model):
    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url

    def run(
        self,
        image,
        prompt: str,
        structured_output_format=None,
        image_name: str = "image.png",
    ):
        b64 = base64.b64encode(image).decode()

        response = requests.post(
            f"{self.base_url}/predict",
            json={
                "image": b64,
                "prompt": prompt,
                "max_new_tokens": 1000,
            },
            timeout=120,
        )
        response.raise_for_status()
        result = response.json()

        return result.get("answer", str(result))
