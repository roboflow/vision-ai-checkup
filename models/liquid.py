import io

from .model import Model


class LiquidVLModel(Model):
    def __init__(self, model_id: str = "LiquidAI/LFM2.5-VL-1.6B"):
        self.model_id = model_id
        self._model = None
        self._processor = None

    def _load(self):
        if self._model is None:
            from transformers import AutoModelForImageTextToText, AutoProcessor

            self._model = AutoModelForImageTextToText.from_pretrained(
                self.model_id,
                device_map="auto",
            )
            self._processor = AutoProcessor.from_pretrained(self.model_id)

    def run(
        self,
        image: bytes,
        prompt: str,
        structured_output_format=None,
        image_name: str = "image.png",
    ):
        from PIL import Image

        self._load()

        pil_image = Image.open(io.BytesIO(image)).convert("RGB")

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        inputs = self._processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            tokenize=True,
        ).to(self._model.device)

        outputs = self._model.generate(**inputs, max_new_tokens=1024)
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        result = self._processor.decode(new_tokens, skip_special_tokens=True)

        return result
