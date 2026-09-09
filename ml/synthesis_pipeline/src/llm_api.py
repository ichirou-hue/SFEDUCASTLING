import json
import logging
import os
from pathlib import Path
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import ValidationError

from src.prompts import SYSTEM_PROMPT, build_user_prompt
from src.schemas import CoachExplanation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")


class ChessLLMClient:
    def __init__(self):
        api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("API-ключ не найден в переменных окружения или файле .env!")

        base_url = os.getenv("OPENAI_BASE_URL", "https://fancy-sky-cdc1.dani4-10.workers.dev/openai/v1")
        self.model_name = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url
        )

    async def generate_coach_explanation(self, fen: str, engine_data: dict) -> CoachExplanation | None:
        """Асинхронно отправляет запрос к LLM и валидирует ответ через схему CoachExplanation."""
        user_prompt = build_user_prompt(
            fen=fen,
            engine_data=engine_data
        )

        content_str = ""
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.2
            )

            if isinstance(response, str):
                content_str = response
            elif hasattr(response, "choices"):
                content_str = response.choices[0].message.content
            elif isinstance(response, dict):
                content_str = response["choices"][0]["message"]["content"]
            else:
                content_str = str(response)

            content_str = content_str.strip()

            if content_str.startswith("```json"):
                content_str = content_str[7:]
            if content_str.startswith("```"):
                content_str = content_str[3:]
            if content_str.endswith("```"):
                content_str = content_str[:-3]
            content_str = content_str.strip()

            raw_data = json.loads(content_str)
            return CoachExplanation(**raw_data)

        except json.JSONDecodeError as e:
            logger.error(f"Модель вернула невалидный JSON: {e}\nСырой ответ: {content_str}")
        except ValidationError as e:
            logger.error(f"Ошибка валидации Pydantic-схемы: {e}")
        except Exception as e:
            logger.error(f"Ошибка API / Сетевой сбой: {e}")

        return None