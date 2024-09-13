from openai import OpenAI

from src.socket_instance import emit_agent
from src.logger import Logger, route_logger
from src.config import Config

logger = Logger()


# I think this is broken if we exceed the token limit. FIXME: look up how to get more than a single response and reconstruct them. 
class OpenAi:
    def __init__(self):
        config = Config()
        api_key = config.get_openai_api_key()
        base_url = config.get_openai_api_base_url()
        self.client = OpenAI(api_key=api_key, base_url=base_url, _strict_response_validation=True)

    def inference(self, model_id: str, prompt: str) -> str:
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt.strip(),
                    }
                ],
                model=model_id,
                temperature=0,
                response_format={"type": "json_object"}
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            # Log the error
            logger.error(f"An error occurred during inference: {str(e)}", exc_info=True)
            # Optionally, you can re-raise the exception or return a fallback response
            raise e  # or return "An error occurred during inference."
