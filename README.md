# SFEDUCASTLING ♟️
**AI Chess Platform based on GigaChat & Maia 2**

### 🎯 Цели проекта (ТЗ)
1. **Описание позиции** и лучший ход (Maia 2).
2. **Поиск мата** в N ходов (N < 6).
3. **Определение стадии игры** и пешечных структур.
4. **Генерация планов** обучения и игры.

### 👥 Команда
* **Егор** — Project Lead (Архитектура, GigaChat API)
* **Митя** — Backend Developer (Python, API)
* **Илья Бабченков** — Frontend Developer (Веб-интерфейс)
* **Илья Слынько** — ML Engineer (Maia 2, Датасеты)
* **Даниил** — ML Engineer (LLM Tuning, Промпты)
* **Тима** — UI/UX Designer (Figma, Тестирование)

### 🛠 Стек
- **LLM:** GigaChat API
- **Engine:** Maia 2 / Stockfish
- **Web:** HTML/JS/CSS

```plaintext
└── 📁 project                                      # корневая директория проекта
    ├── 📁 analysis                                  # модуль шахматного анализа (движки, аннотации, базы данных)
    │   ├── 📁 annotators                            # аннотаторы ошибок и меток NAG
    │   │   ├── 📄 mistake_classifier.py              # классификация ошибок (зевок, неточность, блестящий ход)
    │   │   └── 📄 nag_annotator.py                   # автоматическая простановка меток NAG (!!, ?, ?? и т.д.)
    │   ├── 📁 databases                              # базы данных: дебюты, партии, векторное хранилище
    │   │   ├── 📁 games_db                           # база партий (гроссмейстерские и пользовательские)
    │   │   │   ├── 📄 games.db                       # SQLite-файл с партиями
    │   │   │   └── 📄 models.py                      # ORM-модели для работы с партиями
    │   │   ├── 📁 openings_db                        # база дебютов (ECO-коды, варианты)
    │   │   │   ├── 📄 models.py                      # ORM-модели для дебютов
    │   │   │   └── 📄 openings.db                    # SQLite-файл с дебютами
    │   │   └── 📁 vector_db                          # векторная база данных для поиска похожих позиций
    │   │       ├── 📄 faiss_index                    # индекс FAISS для быстрого поиска эмбеддингов
    │   │       └── 📄 milvus_client.py               # клиент для работы с Milvus (альтернатива FAISS)
    │   ├── 📁 engines                                # шахматные движки
    │   │   ├── 📁 maia                               # Maia – человеко-подобный движок (имитация игры людей)
    │   │   │   ├── 📄 engine.py                      # обёртка для запуска Maia
    │   │   │   └── 📄 predictor.py                   # предсказание хода для заданного рейтинга
    │   │   └── 📁 stockfish                          # Stockfish – классический шахматный движок
    │   │       ├── 📄 depth_config.py                # настройка глубины расчёта
    │   │       ├── 📄 engine.py                      # обёртка для запуска Stockfish
    │   │       └── 📄 evaluator.py                   # оценка позиции, расчёт лучшего хода
    │   └── 📄 similarity_search.py                   # поиск похожих позиций через векторную БД
    ├── 📁 backend                                    # серверная часть (API, очередь, мониторинг)
    │   ├── 📁 api_gateway                            # FastAPI – основной шлюз
    │   │   ├── 📁 routes                             # эндпоинты API
    │   │   │   ├── 📄 analyze.py                     # анализ позиции (Stockfish + Maia + комментарии)
    │   │   │   ├── 📄 arena.py                       # запуск турниров агентов
    │   │   │   ├── 📄 chat.py                        # диалог с ИИ-помощником
    │   │   │   └── 📄 lesson.py                      # получение адаптивных уроков
    │   │   ├── 📄 app.py                             # инициализация FastAPI-приложения
    │   │   ├── 📄 dependencies.py                    # общие зависимости (подключение к БД, загрузка моделей)
    │   │   └── 📄 models.py                          # Pydantic-модели для валидации запросов/ответов
    │   ├── 📁 async_queue                            # асинхронная очередь задач (Celery + Redis)
    │   │   ├── 📄 celery_app.py                      # конфигурация Celery
    │   │   └── 📄 tasks.py                           # определение фоновых задач (тяжёлые вычисления)
    │   └── 📁 monitoring                             # метрики и мониторинг
    │       ├── 📁 grafana_dashboards                  # дашборды Grafana
    │       │   └── 📄 dashboard.json                  # конфигурация дашборда
    │       └── 📄 prometheus_metrics.py               # сбор метрик для Prometheus
    ├── 📁 config                                      # конфигурационные файлы (YAML)
    │   ├── 📄 lora_adapters.yaml                      # настройки LoRA-адаптеров (какие адаптеры загружать)
    │   ├── 📄 model_paths.yaml                        # пути к весам LLM, LLaVA, эмбеддеров
    │   ├── 📄 stockfish_config.yaml                   # параметры Stockfish (глубина, потоки, hash)
    │   └── 📄 tone_profiles.yaml                      # профили тона для дружелюбных объяснений (мотивирующий, спокойный и т.д.)
    ├── 📁 docs                                        # документация
    │   ├── 📁 api                                     # документация API
    │   │   └── 📄 openapi.yaml                        # OpenAPI спецификация (автогенерируемая)
    │   ├── 📁 architecture                            # архитектурные диаграммы
    │   │   └── 📄 diagrams.drawio                     # диаграммы в draw.io
    │   └── 📄 user_guide.md                           # руководство пользователя
    ├── 📁 evaluation                                  # оценка агентов и метрики
    │   ├── 📁 arena                                   # арена для соревнований агентов
    │   │   ├── 📁 agents                              # реализации агентов
    │   │   │   ├── 📄 lora_llm_agent.py               # агент на базе LLM с LoRA-адаптером
    │   │   │   ├── 📄 maia_agent.py                   # агент на базе Maia
    │   │   │   └── 📄 stockfish_agent.py              # агент на базе Stockfish
    │   │   └── 📄 tournament_runner.py                # запуск турниров между агентами, сбор результатов
    │   ├── 📁 metrics                                 # расчёт метрик силы игры
    │   │   ├── 📄 blunder_rate.py                     # частота грубых ошибок
    │   │   ├── 📄 elo_calculator.py                   # расчёт рейтинга Эло
    │   │   └── 📄 move_accuracy.py                    # точность предсказания ходов (сравнение с мастерами)
    │   └── 📁 tracker                                 # трекеры экспериментов
    │       ├── 📄 mlflow_setup.py                     # интеграция с MLflow
    │       └── 📄 wandb_setup.py                      # интеграция с Weights & Biases
    ├── 📁 frontend                                    # клиентская часть (React)
    │   ├── 📁 public                                  # статические файлы
    │   │   ├── 📄 favicon.ico                         # иконка сайта
    │   │   └── 📄 index.html                          # основной HTML-шаблон
    │   ├── 📁 src                                     # исходный код React
    │   │   ├── 📁 api                                 # взаимодействие с бэкендом
    │   │   │   └── 📄 client.js                       # HTTP-клиент (axios/fetch)
    │   │   ├── 📁 components                          # React-компоненты
    │   │   │   ├── 📄 Chessboard.js                   # интерактивная шахматная доска
    │   │   │   ├── 📄 LessonView.js                   # отображение уроков и заданий
    │   │   │   ├── 📄 ReportView.js                   # отображение комментариев, оценок, похожих позиций
    │   │   │   └── 📄 Uploader.js                     # загрузка изображений/FEN/PGN
    │   │   ├── 📁 styles                              # CSS-стили
    │   │   │   ├── 📄 components.css                  # стили компонентов
    │   │   │   └── 📄 main.css                        # общие стили
    │   │   ├── 📄 App.js                              # корневой компонент React
    │   │   └── 📄 index.js                            # точка входа React-приложения
    │   ├── 📄 package.json                            # зависимости и скрипты фронтенда
    │   └── 📄 README.md                               # описание фронтенд-части
    ├── 📁 input_gateway                               # мультимодальный шлюз (обработка разных форматов ввода)
    │   ├── 📁 handlers                                # обработчики типов ввода
    │   │   ├── 📄 fen_parser.py                       # парсинг FEN → объект доски
    │   │   ├── 📄 image_handler.py                    # обработка изображений (PNG/JPG) → передача в vision
    │   │   ├── 📄 pgn_parser.py                       # парсинг PGN → история ходов + метаданные
    │   │   └── 📄 validator.py                        # валидация корректности входных данных
    │   └── 📄 orchestrator.py                         # выбор нужного обработчика на основе типа входа
    ├── 📁 learning_system                             # адаптивная система обучения (Duolingo-стиль)
    │   ├── 📁 adaptive_engine                         # движок подбора контента
    │   │   ├── 📄 content_generator.py                # генерация заданий через LLM
    │   │   ├── 📄 difficulty_adjuster.py              # подстройка сложности под уровень
    │   │   ├── 📄 spaced_repetition.py                # планировщик повторений (Spaced Repetition)
    │   │   └── 📄 topic_selector.py                   # выбор темы на основе слабых мест
    │   ├── 📁 exercises                               # типы упражнений
    │   │   ├── 📄 endgame_practice.py                 # эндшпильные задачи
    │   │   ├── 📄 explain_move.py                     # объяснить, почему ход хорош/плох
    │   │   ├── 📄 find_best_move.py                   # найти лучший ход
    │   │   ├── 📄 opening_quiz.py                     # тесты по дебютам
    │   │   └── 📄 reconstruct_game.py                 # восстановить последовательность ходов
    │   ├── 📁 feedback_loop                           # обратная связь для улучшения учебного плана
    │   │   ├── 📄 curriculum_updater.py               # корректировка дерева навыков на основе данных
    │   │   └── 📄 performance_analytics.py            # аналитика успешности выполнения заданий
    │   ├── 📁 gamification                            # геймификация
    │   │   ├── 📄 achievements.py                     # достижения (бейджи)
    │   │   ├── 📄 leaderboard.py                      # таблица лидеров
    │   │   ├── 📄 streaks.py                          # отслеживание серий (daily streak)
    │   │   └── 📄 xp_system.py                        # начисление очков опыта
    │   └── 📁 user_profile                            # профиль знаний и прогресса
    │       ├── 📄 mastery_tracker.py                  # уровень владения каждой темой (0–100%)
    │       ├── 📄 skill_tree.yaml                     # дерево навыков (дебюты, тактика, эндшпиль...)
    │       ├── 📄 strength_model.py                   # модель сильных сторон пользователя
    │       └── 📄 weakness_model.py                   # модель слабых мест (на основе ошибок)
    ├── 📁 memory                                      # память и RAG
    │   ├── 📁 long_term_memory                        # долговременное хранение истории игр
    │   │   ├── 📄 memory.db                           # SQLite с историей игр пользователя
    │   │   └── 📄 models.py                           # ORM-модели для памяти
    │   ├── 📁 personalized_lessons                    # персонализированные уроки
    │   │   ├── 📄 generator.py                        # генерация уроков на основе истории
    │   │   └── 📄 storage.py                          # хранение сгенерированных уроков
    │   └── 📁 rag                                     # Retrieval-Augmented Generation
    │       ├── 📄 generator.py                        # генерация ответа с использованием LLM + контекста
    │       ├── 📄 prompt_templates.py                 # шаблоны промптов с подстановкой контекста
    │       └── 📄 retriever.py                        # поиск релевантных позиций/уроков в векторной БД
    ├── 📁 perception                                  # «глаза» системы – компьютерное зрение и нормализация
    │   ├── 📁 converters                              # конвертеры между форматами
    │   │   ├── 📄 board_to_fen.py                     # объект доски → FEN
    │   │   ├── 📄 fen_to_board.py                     # FEN → объект доски
    │   │   └── 📄 pgn_to_moves.py                     # PGN → последовательность ходов
    │   ├── 📁 embedder                                # эмбеддер позиций
    │   │   ├── 📄 model.py                            # модель для получения эмбеддинга позиции
    │   │   └── 📄 vector_store.py                     # интерфейс для сохранения/поиска эмбеддингов
    │   ├── 📁 vision                                  # работа с изображениями
    │   │   └── 📁 llava_model                         # LLaVA – мультимодальная модель для распознавания доски
    │   │       ├── 📄 inference.py                    # запуск инференса LLaVA
    │   │       └── 📄 model.py                        # загрузка и инициализация LLaVA
    │   └── 📄 position_normalizer.py                  # приведение позиции к каноническому виду (зеркалирование для чёрных и т.п.)
    ├── 📁 reasoning                                   # «учитель» – LLM, дружелюбные объяснения, персонализация
    │   ├── 📁 assistant                               # диалоговый помощник
    │   │   ├── 📄 dialogue_manager.py                 # управление контекстом разговора
    │   │   └── 📄 explanation_generator.py            # генерация технических объяснений ходов
    │   ├── 📁 commentary                              # комментаторы
    │   │   ├── 📄 commentary_generator.py             # базовый технический комментатор
    │   │   └── 📄 friendly_explanation_engine.py      # дружелюбная версия (адаптация тона и стиля)
    │   ├── 📁 llm_core                                # ядро языковых моделей
    │   │   ├── 📁 base_model                          # базовая мультимодальная модель
    │   │   │   ├── 📄 config.yaml                     # конфигурация модели (размер, путь)
    │   │   │   └── 📄 llava_model.py                  # загрузка и инференс LLaVA (или другой VLM)
    │   │   ├── 📁 lora_adapters                       # LoRA-адаптеры для разных задач
    │   │   │   ├── 📁 commentary_adapter              # адаптер для стиля комментариев
    │   │   │   │   └── 📄 adapter.bin                 # веса адаптера
    │   │   │   ├── 📁 gameplay_adapter                # адаптер для игровой силы
    │   │   │   │   └── 📄 adapter.bin
    │   │   │   ├── 📁 reasoning_adapter               # адаптер для рассуждений о стратегии
    │   │   │   │   └── 📄 adapter.bin
    │   │   │   └── 📁 vision_adapter                  # адаптер для улучшения распознавания доски
    │   │   │       └── 📄 adapter.bin
    │   │   └── 📁 sft_pipeline                        # пайплайн supervised fine-tuning
    │   │       ├── 📄 dataset_loader.py               # загрузка и подготовка датасетов для SFT
    │   │       └── 📄 train.py                        # скрипт дообучения (LoRA или полное)
    │   ├── 📁 personalization                         # настройка тона и стиля под пользователя
    │   │   ├── 📁 template_library                    # библиотека шаблонов фраз
    │   │   │   ├── 📄 mistakes.yaml                   # шаблоны для ошибок
    │   │   │   ├── 📄 praise.yaml                     # шаблоны для похвалы
    │   │   │   └── 📄 teaching.yaml                   # шаблоны для обучающих объяснений
    │   │   ├── 📄 style_personalizer.py               # подстройка языка (метафоры, жаргон)
    │   │   └── 📄 tone_controller.py                  # выбор тона (мотивирующий, спокойный, шутливый)
    │   └── 📄 feedback_integrator.py                  # сбор лайков/дизлайков на комментарии, обновление профиля
    ├── 📁 research                                    # исследовательский блок для публикации ACMMM
    │   ├── 📁 experiment_runner                       # автоматический прогон экспериментов
    │   │   └── 📄 runner.py                           # скрипт для запуска ablation studies
    │   ├── 📁 latex_exporter                          # экспорт результатов в LaTeX
    │   │   └── 📄 exporter.py                         # генерация LaTeX-таблиц и графиков
    │   ├── 📁 paper_assets                            # артефакты для научной статьи
    │   │   ├── 📁 ablation_results                    # результаты ablation studies
    │   │   ├── 📁 datasets                            # датасеты, использованные в статье
    │   │   └── 📁 models                              # веса моделей для воспроизводимости
    │   └── 📁 results_aggregator                      # агрегация результатов
    │       ├── 📄 aggregator.py                       # сбор метрик из разных прогонов
    │       └── 📄 visualizer.py                       # построение графиков и диаграмм
    ├── 📁 scripts                                     # утилиты для автоматизации
    │   ├── 📄 download_datasets.py                    # скрипт для загрузки датасетов (Lichess, Chess.com)
    │   ├── 📄 export_paper.py                         # генерация итогового LaTeX-файла статьи
    │   ├── 📄 run_tournament.py                       # запуск турнира между агентами
    │   └── 📄 train_lora.py                           # запуск дообучения LoRA-адаптеров
    ├── 📁 tests                                       # тестирование
    │   ├── 📁 e2e                                     # сквозные тесты (end‑to‑end)
    │   │   └── 📄 test_user_flow.py                   # тест пользовательского сценария (загрузка, анализ, урок)
    │   ├── 📁 integration                             # интеграционные тесты
    │   │   ├── 📄 test_api.py                         # тесты API-эндпоинтов
    │   │   └── 📄 test_arena.py                       # тесты арены агентов
    │   └── 📁 unit                                    # модульные тесты
    │       ├── 📄 test_embedder.py                    # тесты эмбеддера позиций
    │       └── 📄 test_parsers.py                     # тесты парсеров (FEN, PGN)
    ├── 📁 training_data                               # сбор и подготовка датасетов
    │   ├── 📁 augmentor                               # аугментация данных
    │   │   ├── 📄 fen_augmentation.py                 # аугментация FEN (зеркалирование, перестановки)
    │   │   └── 📄 image_augmentation.py               # аугментация изображений (повороты, обрезания)
    │   ├── 📁 collectors                              # сбор данных из внешних источников
    │   │   ├── 📄 chesscom_api.py                     # загрузка партий с Chess.com API
    │   │   ├── 📄 lichess_api.py                      # загрузка партий с Lichess API
    │   │   └── 📄 pgn_archives.py                     # импорт из локальных PGN-файлов
    │   ├── 📁 labeler                                 # автоматическая разметка
    │   │   ├── 📄 nag_labeler.py                      # простановка NAG-меток (!!, ?, ??)
    │   │   └── 📄 stockfish_labeler.py                # расчёт оценок и лучших ходов через Stockfish
    │   └── 📄 dataset_builder.py                      # формирование итоговых датасетов (для SFT, эмбеддингов)
    ├── 📄 LICENSE                                     # лицензия проекта
    ├── 📄 pyproject.toml                              # зависимости и настройки (poetry/pip)
    └── 📄 README.md                                   # главный файл описания проекта
plaintext```
