# Публичный пакет GigaBench

Архив содержит код, конфигурации, тесты и документацию. Локальные и
замороженные приватные задания, веса моделей и результаты в архив не входят.
Для пробного CPU-прогона из каталога распаковки:

```bash
python -m pip install -r requirements.txt
python run.py all --backend mock --profile smoke --without-legacy \
  --without-explanations --without-commentary-auto --out runs/public-smoke
```

Mock проверяет работу пайплайна, а не качество шахматной модели.
Вложенный `requirements.txt` генерируется из закреплённой версии `chess`
в общем файле зависимостей проекта.
См. [метрики](METRICS.md), [происхождение задач](PROVENANCE.md) и
[валидацию](VALIDATION.md). `PUBLIC_MANIFEST.json` содержит контрольные суммы файлов.
