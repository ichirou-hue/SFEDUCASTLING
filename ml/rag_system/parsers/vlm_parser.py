import os
import torch
import pymupdf
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

def run_vlm_parser(
    pdf_filename="300_shahmatnyh_partiy.pdf",
    output_filename="tarrasch_vlm_clean.txt",
    start_page=5,
    end_page=15
):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pdf_path = os.path.join(base_dir, "data", "raw_books", pdf_filename)
    out_dir = os.path.join(base_dir, "data", "parsed_texts")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, output_filename)

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Файл не найден: {pdf_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[VLM] Инициализация Qwen2-VL на устройстве: {device.upper()}...")

    model_id = "Qwen/Qwen2-VL-2B-Instruct"
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_id)

    doc = pymupdf.open(pdf_path)
    total_pages = len(doc)
    end_page = min(end_page, total_pages)

    prompt = (
        "Распознай русский текст с этой страницы шахматной книги. "
        "Читай по колонкам (сначала левая сверху вниз, затем правая). "
        "Игнорируй картинки диаграмм и мусорные символы. "
        "Склеивай переносы слов. Извлекай связные комментарии к партиям."
    )

    full_text = []

    for p in range(start_page - 1, end_page):
        print(f"-> VLM парсит страницу {p + 1} из {end_page}...", flush=True)
        page = doc[p]
        pix = page.get_pixmap(dpi=200)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text_prompt], images=[img], padding=True, return_tensors="pt").to(device)

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=1500)

        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
        page_output = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        full_text.append(f"--- Страница {p + 1} ---\n" + page_output)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(full_text))

    print(f"[OK] Реальный текст книги успешно распознан VLM и сохранен в: {out_path}")

if __name__ == "__main__":
    run_vlm_parser(start_page=5, end_page=12)