import base64
import anthropic
from .model import Model

ANTHROPIC_TEMPERATURE = 0.1


class AnthropicModel(Model):
    def __init__(self, model_id: str, thinking_budget: int = 0, adaptive_thinking: bool = False):
        self.model_id = model_id
        self.thinking_budget = thinking_budget
        self.adaptive_thinking = adaptive_thinking
        # Increase timeout significantly for extended thinking
        self.client = anthropic.Anthropic(timeout=None if (thinking_budget > 0 or adaptive_thinking) else 60.0)

    def run(
        self,
        image: str,
        prompt: str,
        image_name="image.png",
        structured_output_format=None,
    ):
        kwargs = {
            "model": self.model_id,
            "max_tokens": 1024,
            "temperature": ANTHROPIC_TEMPERATURE,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/"
                                + image_name.split(".")[-1].replace("jpg", "jpeg"),
                                "data": base64.b64encode(image).decode(),
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                }
            ],
        }

        if self.adaptive_thinking:
            kwargs["thinking"] = {"type": "adaptive"}
            kwargs.pop("temperature", None)
            kwargs["max_tokens"] = 16000
        elif self.thinking_budget > 0:
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": self.thinking_budget}
            kwargs["temperature"] = 1.0
            # Ensure max_tokens is greater than thinking budget
            kwargs["max_tokens"] = self.thinking_budget + 2048

        if structured_output_format:
            # Use tool use to enforce JSON structure, which works with extended thinking (unlike prefill)
            tool_name = "submit_result"
            tool = {
                "name": tool_name,
                "description": "Submit the final answer in JSON format.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "answer_object": {
                            "type": "object",
                            "description": "The answer if it is a JSON object (dictionary)."
                        },
                        "answer_list": {
                            "type": "array",
                            "description": "The answer if it is a JSON list (array)."
                        }
                    }
                    # Removed oneOf logic as it is not supported at top level
                }
            }
            kwargs["tools"] = [tool]
            kwargs["tool_choice"] = {"type": "auto"}
            
            # Append instruction to prompt
            prompt += f"\n\nIMPORTANT: You MUST use the '{tool_name}' tool to submit your final result. Do not output the JSON as plain text."
            
            # Update the prompt in the message
            kwargs["messages"][0]["content"][1]["text"] = prompt

        message = self.client.messages.create(**kwargs)

        # Handle tool use response
        if structured_output_format:
            for block in message.content:
                if block.type == "tool_use" and block.name == tool_name:
                    import json
                    # Return the tool input as a JSON string
                    input_data = block.input
                    if "answer_object" in input_data:
                        return json.dumps(input_data["answer_object"])
                    elif "answer_list" in input_data:
                         return json.dumps(input_data["answer_list"])
                    else:
                        # Fallback if model puts data elsewhere or flattened
                        return json.dumps(input_data)

        # Handle multiple content blocks (e.g. thinking block + text block)
        text_content = ""
        for block in message.content:
            if block.type == "text":
                text_content += block.text
        
        return text_content
