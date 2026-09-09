import json
import re
import time
import requests
from requests.adapters import HTTPAdapter
import urllib3
from rich.console import Console

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
console = Console()


def strip_thinking(text: str) -> str:
    """Удаляет блок <think>...</think> или неполный <think> тег из вывода модели."""
    # Если есть закрытый блок <think>...</think>, удаляем его целиком
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    
    # Если модель оборвалась до закрытия </think>, отрезаем всё до конца тега
    if "<think>" in cleaned and "</think>" not in cleaned:
        cleaned = re.sub(r'<think>.*', '', cleaned, flags=re.DOTALL).strip()
        
    # Если после очистки осталась пустота, возвращаем текст без тегов
    if not cleaned:
        cleaned = text.replace("<think>", "").replace("</think>", "").strip()
        
    return cleaned


class HTTP11Adapter(HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        kwargs['ssl_version'] = None
        return super().init_poolmanager(*args, **kwargs)


class CloudGigaChessModel:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        
        # КРИТИЧНО: Отключаем подхват системных переменных HTTP_PROXY / HTTPS_PROXY
        # Запросы в Cloud.ru идут напрямую, не конфликтуя с W&B прокси
        self.session.trust_env = False
        
        self.session.mount("https://", HTTP11Adapter())
        self._warmup()

    def _warmup(self, max_attempts: int = 40, delay: int = 10):
        ping_url = f"{self.base_url}/ping"
        console.print(f"[bold yellow]⏳ Проверка статуса Model RUN инстанса...[/] [dim]({ping_url})[/]")
        
        for attempt in range(1, max_attempts + 1):
            try:
                resp = self.session.get(ping_url, timeout=15, verify=False)
                if resp.status_code == 200:
                    console.print(f"[bold green]✔ Инстанс активен и готов к приёму запросов![/]\n")
                    return
            except Exception:
                pass
            
            console.print(f"  [dim]Попытка {attempt}/{max_attempts}: инстанс прогревается (cold start)... ждём {delay} сек[/]")
            time.sleep(delay)

        console.print(f"[bold red]Предупреждение:[/] Инстанс не ответил на /ping за отведённое время, пробуем слать запросы напрямую.")

    def generate(self, prompt: str, fen: str = "") -> str:
        url = f"{self.base_url}/chat" if not self.base_url.endswith("/chat") else self.base_url

        message = {
            "role": "user",
            "content": prompt
        }
        if fen:
            message["attachments"] = [fen]

        payload = {
            "messages": [message],
            "temperature": 0.1,
            "top_p": 0.95,
            "max_tokens": 256,
            "n": 1,
            "repetition_penalty": 1.0,
            "model": "gigachess"
        }

        try:
            response = self.session.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"},
                verify=False,
                timeout=300
            )
            response.raise_for_status()
            data = response.json()
            raw_text = data["choices"][0]["message"]["content"].strip()
            return strip_thinking(raw_text)
        except Exception as e:
            return f"Error: {e}"


class LocalChessModel:
    def __init__(self, model_path: str, base_model_path: str = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        console.print(f"Загрузка локальной модели: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_path if base_model_path else model_path,
            trust_remote_code=True
        )
        
        if base_model_path:
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
            self.model = PeftModel.from_pretrained(base_model, model_path)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
        self.model.eval()

    def generate(self, prompt: str, fen: str = "") -> str:
        import torch
        
        full_content = f"FEN: {fen}\n{prompt}" if fen else prompt
        
        messages = [
            {"role": "system", "content": "You are a concise chess engine and assistant. Do not use reasoning traces or <think> tags. Output the final answer directly."},
            {"role": "user", "content": full_content}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
        generated_ids = [
            output_ids[len(input_ids):] 
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        raw_response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return strip_thinking(raw_response)


def init_model_runner(model_path: str, base_model_path: str = None):
    if model_path.startswith("http://") or model_path.startswith("https://"):
        return CloudGigaChessModel(model_path)
    return LocalChessModel(model_path, base_model_path)