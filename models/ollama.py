import base64

import requests

from .model import Model


class OllamaModel(Model):
    def __init__(self, model_id: str, base_url: str):
        self.model_id = model_id
        self.base_url = base_url.rstrip("/")

    def run(
        self,
        image: str,
        prompt: str,
        structured_output_format=None,
        image_name: str = "image.png",
    ):
        response = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model_id,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                        "images": [base64.b64encode(image).decode()],
                    }
                ],
                "stream": False,
                "think": False,
                "options": {"num_predict": 1024},
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["message"]["content"]
