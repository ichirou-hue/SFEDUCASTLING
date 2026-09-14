"""Стандартное содержимое учебного курса SFEDUCASTLING.

Этот файл содержит только декларативные данные курса. Он не обращается к БД,
поэтому его удобно расширять, ревьюить и тестировать отдельно.

Типы заданий:
- select_squares: отметить набор клеток на доске;
- make_move: сделать ход на доске;
- choose_option: выбрать один текстовый вариант ответа.
"""

MODULES = [
    {
        "slug": "pawn",
        "title": "Пешка",
        "description": "Обычный ход, первый ход на две клетки, взятие и ограничения движения пешки.",
        "sort_order": 1,
        "enabled": True,
    },
    {
        "slug": "knight",
        "title": "Конь",
        "description": "Движение буквой «Г», перепрыгивание фигур и взятие конём.",
        "sort_order": 2,
        "enabled": True,
    },
    {
        "slug": "bishop",
        "title": "Слон",
        "description": "Движение по диагоналям, препятствия, цвет полей и взятия.",
        "sort_order": 3,
        "enabled": True,
    },
    {
        "slug": "rook",
        "title": "Ладья",
        "description": "Движение по вертикалям и горизонталям, препятствия и взятия.",
        "sort_order": 4,
        "enabled": True,
    },
    {
        "slug": "queen",
        "title": "Ферзь",
        "description": "Совмещение возможностей ладьи и слона, препятствия и взятия.",
        "sort_order": 5,
        "enabled": True,
    },
    {
        "slug": "king",
        "title": "Король",
        "description": "Движение короля, атакованные поля, взятия и взаимодействие королей.",
        "sort_order": 6,
        "enabled": True,
    },
    {
        "slug": "special-rules",
        "title": "Специальные правила",
        "description": "Рокировка, взятие на проходе и превращение пешки.",
        "sort_order": 7,
        "enabled": True,
    },
    {
        "slug": "check-mate-stalemate",
        "title": "Шах, мат и пат",
        "description": "Шах, способы защиты, мат и пат как основные завершения позиции.",
        "sort_order": 8,
        "enabled": True,
    },
]


def _options(*pairs: tuple[str, str]) -> list[dict[str, str]]:
    return [{"id": option_id, "label": label} for option_id, label in pairs]


PAWN_LESSONS = [
    {
        "slug": "pawn-basic-movement",
        "title": "Обычный ход пешки",
        "theory": (
            "Пешка ходит только вперёд по своей вертикали. Обычно она перемещается на одну клетку. "
            "Белые пешки движутся в сторону восьмой горизонтали, чёрные — в сторону первой."
        ),
        "sort_order": 1,
        "tasks": [
            {
                "task_type": "select_squares",
                "title": "Пешка в начальной позиции",
                "instruction": "Отметьте все клетки, на которые сейчас может пойти белая пешка e2.",
                "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                "source_square": "e2",
                "difficulty": 1,
                "payload": {"piece": "P", "mode": "all_legal_moves"},
                "explanation": "Из начального положения пешка e2 может пойти на e3 или сразу на e4.",
                "sort_order": 1,
            },
            {
                "task_type": "make_move",
                "title": "Сделайте первый ход пешкой",
                "instruction": "Сделайте любой допустимый ход белой пешкой с e2.",
                "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                "source_square": "e2",
                "difficulty": 1,
                "payload": {"piece": "P", "mode": "any_legal_move"},
                "explanation": "На первом ходу пешка может пройти одну клетку e2-e3 или две клетки e2-e4.",
                "sort_order": 2,
            },
        ],
    },
    {
        "slug": "pawn-after-first-move",
        "title": "После первого хода",
        "theory": (
            "Право пройти сразу две клетки есть только из начального положения. После того как пешка "
            "сдвинулась, обычный ход вперёд составляет ровно одну клетку. Назад пешка не ходит."
        ),
        "sort_order": 2,
        "tasks": [
            {
                "task_type": "select_squares",
                "title": "Пешка уже покинула стартовую клетку",
                "instruction": "Отметьте все клетки, на которые может пойти белая пешка e4.",
                "fen": "7k/8/8/8/4P3/8/8/K7 w - - 0 1",
                "source_square": "e4",
                "difficulty": 1,
                "payload": {"piece": "P", "mode": "all_legal_moves"},
                "explanation": "Пешка с e4 уже не может идти на две клетки: её обычный ход — e4-e5.",
                "sort_order": 1,
            },
            {
                "task_type": "choose_option",
                "title": "Направление движения",
                "instruction": "Как белая пешка ходит без взятия?",
                "fen": "7k/8/8/8/4P3/8/8/K7 w - - 0 1",
                "source_square": "e4",
                "difficulty": 1,
                "payload": {
                    "options": _options(
                        ("forward", "Только вперёд по своей вертикали"),
                        ("backward", "Вперёд и назад по вертикали"),
                        ("diagonal", "Только по диагонали"),
                    ),
                    "correct_option": "forward",
                },
                "explanation": "Без взятия белая пешка движется только вперёд по своей вертикали.",
                "sort_order": 2,
            },
        ],
    },
    {
        "slug": "pawn-capture",
        "title": "Взятие пешкой",
        "theory": (
            "Пешка ходит вперёд, но берёт фигуры противника иначе: на одну клетку по диагонали вперёд. "
            "Белая пешка e4, например, может брать фигуры на d5 и f5."
        ),
        "sort_order": 3,
        "tasks": [
            {
                "task_type": "select_squares",
                "title": "Два возможных взятия",
                "instruction": "Отметьте клетки, на которых белая пешка e4 может взять фигуру противника.",
                "fen": "7k/8/8/3p1p2/4P3/8/8/K7 w - - 0 1",
                "source_square": "e4",
                "difficulty": 1,
                "payload": {"piece": "P", "mode": "capture_squares"},
                "explanation": "Пешка e4 может взять фигуры на d5 и f5.",
                "sort_order": 1,
            },
            {
                "task_type": "make_move",
                "title": "Выполните взятие пешкой",
                "instruction": "Сделайте белой пешкой e4 любое доступное взятие.",
                "fen": "7k/8/8/3p1p2/4P3/8/8/K7 w - - 0 1",
                "source_square": "e4",
                "difficulty": 1,
                "payload": {"piece": "P", "mode": "legal_capture"},
                "explanation": "Корректным будет e4xd5 или e4xf5: пешка берёт по диагонали вперёд.",
                "sort_order": 2,
            },
        ],
    },
    {
        "slug": "pawn-blocking",
        "title": "Препятствия перед пешкой",
        "theory": (
            "Пешка не может перепрыгнуть фигуру и не может взять фигуру, стоящую прямо перед ней. "
            "Если путь вперёд закрыт, возможны только допустимые диагональные взятия."
        ),
        "sort_order": 4,
        "tasks": [
            {
                "task_type": "select_squares",
                "title": "Путь вперёд закрыт",
                "instruction": "Отметьте все легальные клетки для белой пешки d4.",
                "fen": "7k/8/8/2ppp3/3P4/8/8/K7 w - - 0 1",
                "source_square": "d4",
                "difficulty": 2,
                "payload": {"piece": "P", "mode": "all_legal_moves"},
                "explanation": "Поле d5 занято, поэтому пешка не идёт вперёд, но может взять на c5 или e5.",
                "sort_order": 1,
            },
            {
                "task_type": "choose_option",
                "title": "Можно ли перепрыгнуть препятствие?",
                "instruction": "Что происходит, если прямо перед пешкой стоит фигура?",
                "fen": "7k/8/8/3p4/3P4/8/8/K7 w - - 0 1",
                "source_square": "d4",
                "difficulty": 1,
                "payload": {
                    "options": _options(
                        ("blocked", "Пешка не может пройти через эту фигуру"),
                        ("capture_forward", "Пешка берёт фигуру прямо перед собой"),
                        ("jump", "Пешка перепрыгивает её"),
                    ),
                    "correct_option": "blocked",
                },
                "explanation": "Фигура прямо перед пешкой блокирует её движение по вертикали.",
                "sort_order": 2,
            },
        ],
    },
    {
        "slug": "pawn-review",
        "title": "Закрепление: пешка",
        "theory": (
            "Вспомните главное: пешка ходит вперёд, с начальной клетки может пройти две клетки, "
            "берёт по диагонали и не перепрыгивает фигуры."
        ),
        "sort_order": 5,
        "tasks": [
            {
                "task_type": "select_squares",
                "title": "Все возможности пешки",
                "instruction": "Отметьте все легальные клетки для белой пешки d4.",
                "fen": "7k/8/8/2p1p3/3P4/8/8/K7 w - - 0 1",
                "source_square": "d4",
                "difficulty": 2,
                "payload": {"piece": "P", "mode": "all_legal_moves"},
                "explanation": "Пешка может пойти на d5 либо взять фигуру на c5 или e5.",
                "sort_order": 1,
            }
        ],
    },
]


KNIGHT_LESSONS = [
    {
        "slug": "knight-movement",
        "title": "Как ходит конь",
        "theory": (
            "Конь перемещается буквой «Г»: на две клетки по вертикали или горизонтали "
            "и затем на одну клетку перпендикулярно. Из центра доски у коня может быть "
            "до восьми вариантов хода."
        ),
        "sort_order": 1,
        "tasks": [
            {
                "task_type": "select_squares",
                "title": "Конь в центре доски",
                "instruction": "Отметьте все клетки, на которые может перейти белый конь с e4.",
                "fen": "7k/8/8/8/4N3/8/8/K7 w - - 0 1",
                "source_square": "e4",
                "difficulty": 1,
                "payload": {"piece": "N", "mode": "all_legal_moves"},
                "explanation": "С e4 конь может перейти на c3, c5, d2, d6, f2, f6, g3 и g5.",
                "sort_order": 1,
            },
            {
                "task_type": "make_move",
                "title": "Первый ход конём",
                "instruction": "Сделайте любой допустимый ход белым конём с g1.",
                "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                "source_square": "g1",
                "difficulty": 1,
                "payload": {"piece": "N", "mode": "any_legal_move"},
                "explanation": "В начальной позиции конь g1 может пойти на f3 или h3.",
                "sort_order": 2,
            },
        ],
    },
    {
        "slug": "knight-edge",
        "title": "Конь у края доски",
        "theory": (
            "Чем ближе конь к краю или углу доски, тем меньше клеток ему доступно. "
            "Поэтому конь обычно активнее ближе к центру."
        ),
        "sort_order": 2,
        "tasks": [
            {
                "task_type": "select_squares",
                "title": "Конь в углу a1",
                "instruction": "Отметьте все клетки, доступные белому коню с a1.",
                "fen": "7k/8/8/8/8/8/8/N3K3 w - - 0 1",
                "source_square": "a1",
                "difficulty": 1,
                "payload": {"piece": "N", "mode": "all_legal_moves"},
                "explanation": "Из угла a1 коню доступны только b3 и c2.",
                "sort_order": 1,
            },
            {
                "task_type": "select_squares",
                "title": "Конь в углу h8",
                "instruction": "Отметьте все клетки, доступные белому коню с h8.",
                "fen": "k6N/8/8/8/8/8/8/4K3 w - - 0 1",
                "source_square": "h8",
                "difficulty": 1,
                "payload": {"piece": "N", "mode": "all_legal_moves"},
                "explanation": "Из угла h8 коню доступны только f7 и g6.",
                "sort_order": 2,
            },
        ],
    },
    {
        "slug": "knight-jump",
        "title": "Перепрыгивание через фигуры",
        "theory": (
            "Конь — единственная шахматная фигура, которая может перепрыгивать через другие фигуры. "
            "Фигуры рядом с ним не блокируют ход, но конечная клетка не может быть занята своей фигурой."
        ),
        "sort_order": 3,
        "tasks": [
            {
                "task_type": "select_squares",
                "title": "Окружённый конь",
                "instruction": "Конь d4 окружён своими пешками. Отметьте все клетки, на которые он всё равно может перейти.",
                "fen": "7k/8/8/2PPP3/2PNP3/2PPP3/8/K7 w - - 0 1",
                "source_square": "d4",
                "difficulty": 2,
                "payload": {"piece": "N", "mode": "all_legal_moves"},
                "explanation": "Соседние пешки не мешают коню: он перепрыгивает через них.",
                "sort_order": 1,
            },
            {
                "task_type": "make_move",
                "title": "Перепрыгните окружение",
                "instruction": "Сделайте любой допустимый ход конём с d4, не обращая внимания на окружающие пешки.",
                "fen": "7k/8/8/2PPP3/2PNP3/2PPP3/8/K7 w - - 0 1",
                "source_square": "d4",
                "difficulty": 2,
                "payload": {"piece": "N", "mode": "any_legal_move"},
                "explanation": "Конь может перепрыгнуть через окружающие его фигуры.",
                "sort_order": 2,
            },
        ],
    },
    {
        "slug": "knight-capture",
        "title": "Взятие конём",
        "theory": (
            "Конь берёт фигуру противника на той клетке, на которую приходит своим обычным ходом буквой «Г». "
            "Перепрыгиваемые фигуры при этом не снимаются с доски."
        ),
        "sort_order": 4,
        "tasks": [
            {
                "task_type": "select_squares",
                "title": "Найдите доступные взятия",
                "instruction": "Отметьте клетки, на которых белый конь e4 может взять фигуру противника.",
                "fen": "7k/8/8/2r3b1/4N3/8/8/K7 w - - 0 1",
                "source_square": "e4",
                "difficulty": 2,
                "payload": {"piece": "N", "mode": "capture_squares"},
                "explanation": "Конь e4 может взять ладью на c5 и слона на g5.",
                "sort_order": 1,
            },
            {
                "task_type": "make_move",
                "title": "Выполните взятие",
                "instruction": "Сделайте конём e4 любое доступное взятие.",
                "fen": "7k/8/8/2r3b1/4N3/8/8/K7 w - - 0 1",
                "source_square": "e4",
                "difficulty": 2,
                "payload": {"piece": "N", "mode": "legal_capture"},
                "explanation": "Правильным будет Nxc5 или Nxg5.",
                "sort_order": 2,
            },
        ],
    },
    {
        "slug": "knight-review",
        "title": "Закрепление: конь",
        "theory": (
            "В итоговых заданиях вспомните форму хода коня, влияние края доски, "
            "возможность перепрыгивания и правила взятия."
        ),
        "sort_order": 5,
        "tasks": [
            {
                "task_type": "select_squares",
                "title": "Конь после первых ходов",
                "instruction": "Отметьте все легальные клетки для белого коня с g1 в этой позиции.",
                "fen": "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
                "source_square": "g1",
                "difficulty": 2,
                "payload": {"piece": "N", "mode": "all_legal_moves"},
                "explanation": "После e2-e4 коню g1 доступны e2, f3 и h3.",
                "sort_order": 1,
            }
        ],
    },
]


BISHOP_LESSONS = [
    {
        "slug": "bishop-movement",
        "title": "Как ходит слон",
        "theory": (
            "Слон перемещается на любое число свободных клеток по диагонали. Он не может менять цвет полей: "
            "слон, начавший на светлом поле, всю партию остаётся на светлых полях."
        ),
        "sort_order": 1,
        "tasks": [
            {
                "task_type": "select_squares",
                "title": "Слон в центре",
                "instruction": "Отметьте все клетки, на которые может перейти белый слон d4.",
                "fen": "K7/7k/8/8/3B4/8/8/8 w - - 0 1",
                "source_square": "d4",
                "difficulty": 1,
                "payload": {"piece": "B", "mode": "all_legal_moves"},
                "explanation": "Слон d4 движется по четырём диагоналям до края доски.",
                "sort_order": 1,
            },
            {
                "task_type": "make_move",
                "title": "Сделайте ход слоном",
                "instruction": "Сделайте любой допустимый ход белым слоном с d4.",
                "fen": "K7/7k/8/8/3B4/8/8/8 w - - 0 1",
                "source_square": "d4",
                "difficulty": 1,
                "payload": {"piece": "B", "mode": "any_legal_move"},
                "explanation": "Любой ход слона должен оставаться на диагонали от d4.",
                "sort_order": 2,
            },
        ],
    },
    {
        "slug": "bishop-blockers",
        "title": "Препятствия на диагонали",
        "theory": (
            "Слон не умеет перепрыгивать фигуры. Своя фигура останавливает его движение, а фигуру противника "
            "можно взять, но двигаться дальше за неё нельзя."
        ),
        "sort_order": 2,
        "tasks": [
            {
                "task_type": "select_squares",
                "title": "Слон между своими пешками",
                "instruction": "Отметьте все легальные клетки для слона d4.",
                "fen": "K6k/8/8/2P1P3/3B4/8/8/8 w - - 0 1",
                "source_square": "d4",
                "difficulty": 2,
                "payload": {"piece": "B", "mode": "all_legal_moves"},
                "explanation": "Пешки c5 и e5 закрывают верхние диагонали; вниз слон продолжает движение до края.",
                "sort_order": 1,
            },
            {
                "task_type": "choose_option",
                "title": "Может ли слон перепрыгивать?",
                "instruction": "Что происходит, если на диагонали слона стоит своя фигура?",
                "fen": "K7/7k/8/2P5/3B4/8/8/8 w - - 0 1",
                "source_square": "d4",
                "difficulty": 1,
                "payload": {
                    "options": _options(
                        ("stop", "Слон останавливается перед своей фигурой"),
                        ("jump", "Слон перепрыгивает её"),
                        ("capture", "Слон может взять свою фигуру"),
                    ),
                    "correct_option": "stop",
                },
                "explanation": "Собственные фигуры блокируют линию слона.",
                "sort_order": 2,
            },
        ],
    },
    {
        "slug": "bishop-capture",
        "title": "Взятие слоном",
        "theory": (
            "Слон берёт фигуру противника, если она находится на его диагонали и между ними нет других фигур."
        ),
        "sort_order": 3,
        "tasks": [
            {
                "task_type": "select_squares",
                "title": "Две цели на диагоналях",
                "instruction": "Отметьте клетки, на которых слон d4 может взять фигуру противника.",
                "fen": "7k/8/1r3n2/8/3B4/8/8/K7 w - - 0 1",
                "source_square": "d4",
                "difficulty": 2,
                "payload": {"piece": "B", "mode": "capture_squares"},
                "explanation": "Слон может взять ладью b6 и коня f6.",
                "sort_order": 1,
            },
            {
                "task_type": "make_move",
                "title": "Выполните диагональное взятие",
                "instruction": "Сделайте слоном d4 любое доступное взятие.",
                "fen": "7k/8/1r3n2/8/3B4/8/8/K7 w - - 0 1",
                "source_square": "d4",
                "difficulty": 2,
                "payload": {"piece": "B", "mode": "legal_capture"},
                "explanation": "Слон может взять фигуру на b6 или f6.",
                "sort_order": 2,
            },
        ],
    },
    {
        "slug": "bishop-review",
        "title": "Закрепление: слон",
        "theory": "Слон всегда движется по диагонали, не перепрыгивает фигуры и сохраняет цвет своих полей.",
        "sort_order": 4,
        "tasks": [
            {
                "task_type": "choose_option",
                "title": "Цвет полей слона",
                "instruction": "Может ли слон в течение партии перейти с белых полей на чёрные?",
                "fen": "K7/7k/8/8/3B4/8/8/8 w - - 0 1",
                "source_square": "d4",
                "difficulty": 1,
                "payload": {
                    "options": _options(("no", "Нет"), ("yes", "Да")),
                    "correct_option": "no",
                },
                "explanation": "Нет. Диагональное движение всегда сохраняет цвет поля.",
                "sort_order": 1,
            }
        ],
    },
]


ROOK_LESSONS = [
    {
        "slug": "rook-movement",
        "title": "Как ходит ладья",
        "theory": "Ладья ходит на любое число свободных клеток по вертикали или горизонтали.",
        "sort_order": 1,
        "tasks": [
            {
                "task_type": "select_squares",
                "title": "Ладья в центре",
                "instruction": "Отметьте все клетки, на которые может перейти белая ладья d4.",
                "fen": "K6k/8/8/8/3R4/8/8/8 w - - 0 1",
                "source_square": "d4",
                "difficulty": 1,
                "payload": {"piece": "R", "mode": "all_legal_moves"},
                "explanation": "Ладья d4 ходит по четвёртой горизонтали и вертикали d.",
                "sort_order": 1,
            },
            {
                "task_type": "make_move",
                "title": "Сделайте ход ладьёй",
                "instruction": "Сделайте любой допустимый ход белой ладьёй с d4.",
                "fen": "K6k/8/8/8/3R4/8/8/8 w - - 0 1",
                "source_square": "d4",
                "difficulty": 1,
                "payload": {"piece": "R", "mode": "any_legal_move"},
                "explanation": "Ладья может двигаться только прямо: по горизонтали или вертикали.",
                "sort_order": 2,
            },
        ],
    },
    {
        "slug": "rook-blockers",
        "title": "Препятствия для ладьи",
        "theory": "Ладья не перепрыгивает через фигуры. Свои фигуры ограничивают её линию движения.",
        "sort_order": 2,
        "tasks": [
            {
                "task_type": "select_squares",
                "title": "Ладья и свои фигуры",
                "instruction": "Отметьте все легальные клетки для ладьи d4.",
                "fen": "K6k/8/3P4/8/3R1P2/8/8/8 w - - 0 1",
                "source_square": "d4",
                "difficulty": 2,
                "payload": {"piece": "R", "mode": "all_legal_moves"},
                "explanation": "Пешки d6 и f4 ограничивают движение ладьи вверх и вправо.",
                "sort_order": 1,
            },
            {
                "task_type": "choose_option",
                "title": "Линия ладьи",
                "instruction": "Может ли ладья перепрыгнуть через фигуру на своей вертикали?",
                "fen": "K6k/8/3P4/8/3R4/8/8/8 w - - 0 1",
                "source_square": "d4",
                "difficulty": 1,
                "payload": {
                    "options": _options(("no", "Нет"), ("yes", "Да")),
                    "correct_option": "no",
                },
                "explanation": "Нет. Любая фигура на линии останавливает движение ладьи дальше по этой линии.",
                "sort_order": 2,
            },
        ],
    },
    {
        "slug": "rook-capture",
        "title": "Взятие ладьёй",
        "theory": "Ладья может взять первую фигуру противника на своей вертикали или горизонтали.",
        "sort_order": 3,
        "tasks": [
            {
                "task_type": "select_squares",
                "title": "Цели ладьи",
                "instruction": "Отметьте клетки, на которых ладья d4 может взять фигуру противника.",
                "fen": "7k/8/3r4/8/3R2b1/8/8/K7 w - - 0 1",
                "source_square": "d4",
                "difficulty": 2,
                "payload": {"piece": "R", "mode": "capture_squares"},
                "explanation": "Ладья может взять фигуру на d6 или g4.",
                "sort_order": 1,
            },
            {
                "task_type": "make_move",
                "title": "Выполните взятие ладьёй",
                "instruction": "Сделайте ладьёй d4 любое доступное взятие.",
                "fen": "7k/8/3r4/8/3R2b1/8/8/K7 w - - 0 1",
                "source_square": "d4",
                "difficulty": 2,
                "payload": {"piece": "R", "mode": "legal_capture"},
                "explanation": "Правильным будет Rxd6 или Rxg4.",
                "sort_order": 2,
            },
        ],
    },
    {
        "slug": "rook-review",
        "title": "Закрепление: ладья",
        "theory": "Ладья сильнее всего действует по открытым вертикалям и горизонталям, где ей не мешают другие фигуры.",
        "sort_order": 4,
        "tasks": [
            {
                "task_type": "choose_option",
                "title": "Направления ладьи",
                "instruction": "Какие направления доступны ладье?",
                "fen": "K6k/8/8/8/3R4/8/8/8 w - - 0 1",
                "source_square": "d4",
                "difficulty": 1,
                "payload": {
                    "options": _options(
                        ("straight", "Вертикали и горизонтали"),
                        ("diagonal", "Только диагонали"),
                        ("knight", "Ход буквой «Г»"),
                    ),
                    "correct_option": "straight",
                },
                "explanation": "Ладья перемещается только по вертикали и горизонтали.",
                "sort_order": 1,
            }
        ],
    },
]


QUEEN_LESSONS = [
    {
        "slug": "queen-movement",
        "title": "Как ходит ферзь",
        "theory": (
            "Ферзь объединяет движения ладьи и слона: он может идти на любое число свободных клеток "
            "по вертикали, горизонтали или диагонали."
        ),
        "sort_order": 1,
        "tasks": [
            {
                "task_type": "select_squares",
                "title": "Ферзь в центре",
                "instruction": "Отметьте все клетки, на которые может перейти белый ферзь d4.",
                "fen": "K7/7k/8/8/3Q4/8/8/8 w - - 0 1",
                "source_square": "d4",
                "difficulty": 2,
                "payload": {"piece": "Q", "mode": "all_legal_moves"},
                "explanation": "Ферзь d4 контролирует линии ладьи и диагонали слона.",
                "sort_order": 1,
            },
            {
                "task_type": "make_move",
                "title": "Сделайте ход ферзём",
                "instruction": "Сделайте любой допустимый ход белым ферзём с d4.",
                "fen": "K7/7k/8/8/3Q4/8/8/8 w - - 0 1",
                "source_square": "d4",
                "difficulty": 1,
                "payload": {"piece": "Q", "mode": "any_legal_move"},
                "explanation": "Ферзь может двигаться прямо или по диагонали.",
                "sort_order": 2,
            },
        ],
    },
    {
        "slug": "queen-blockers",
        "title": "Препятствия для ферзя",
        "theory": "Несмотря на большой радиус действия, ферзь, как ладья и слон, не умеет перепрыгивать фигуры.",
        "sort_order": 2,
        "tasks": [
            {
                "task_type": "select_squares",
                "title": "Ферзь среди своих фигур",
                "instruction": "Отметьте все легальные клетки для ферзя d4.",
                "fen": "7k/8/3P4/2P1P3/3Q4/2P1P3/8/K7 w - - 0 1",
                "source_square": "d4",
                "difficulty": 2,
                "payload": {"piece": "Q", "mode": "all_legal_moves"},
                "explanation": "Свои фигуры закрывают часть линий ферзя, и за них он пройти не может.",
                "sort_order": 1,
            },
            {
                "task_type": "choose_option",
                "title": "Что умеет ферзь?",
                "instruction": "Как точнее всего описать движение ферзя?",
                "fen": "K7/7k/8/8/3Q4/8/8/8 w - - 0 1",
                "source_square": "d4",
                "difficulty": 1,
                "payload": {
                    "options": _options(
                        ("rook_bishop", "Как ладья и слон вместе"),
                        ("knight", "Как конь"),
                        ("king", "Только на одну клетку"),
                    ),
                    "correct_option": "rook_bishop",
                },
                "explanation": "Ферзь объединяет прямолинейные ходы ладьи и диагональные ходы слона.",
                "sort_order": 2,
            },
        ],
    },
    {
        "slug": "queen-capture",
        "title": "Взятие ферзём",
        "theory": "Ферзь берёт первую фигуру противника на доступной вертикали, горизонтали или диагонали.",
        "sort_order": 3,
        "tasks": [
            {
                "task_type": "select_squares",
                "title": "Цели ферзя",
                "instruction": "Отметьте клетки, на которых ферзь d4 может взять фигуру противника.",
                "fen": "7k/8/1r1p1n2/8/3Q2b1/8/8/K7 w - - 0 1",
                "source_square": "d4",
                "difficulty": 2,
                "payload": {"piece": "Q", "mode": "capture_squares"},
                "explanation": "Ферзь может взять фигуры на b6, d6, f6 и g4.",
                "sort_order": 1,
            },
            {
                "task_type": "make_move",
                "title": "Выполните взятие ферзём",
                "instruction": "Сделайте ферзём d4 любое доступное взятие.",
                "fen": "7k/8/1r1p1n2/8/3Q2b1/8/8/K7 w - - 0 1",
                "source_square": "d4",
                "difficulty": 2,
                "payload": {"piece": "Q", "mode": "legal_capture"},
                "explanation": "Ферзь может выполнить одно из доступных взятий по линии или диагонали.",
                "sort_order": 2,
            },
        ],
    },
    {
        "slug": "queen-review",
        "title": "Закрепление: ферзь",
        "theory": "Ферзь очень подвижен, но подчиняется тем же ограничениям линий, что ладья и слон.",
        "sort_order": 4,
        "tasks": [
            {
                "task_type": "choose_option",
                "title": "Сильная сторона ферзя",
                "instruction": "Почему ферзь контролирует так много клеток?",
                "fen": "K7/7k/8/8/3Q4/8/8/8 w - - 0 1",
                "source_square": "d4",
                "difficulty": 1,
                "payload": {
                    "options": _options(
                        ("combined", "Он ходит и прямо, и по диагонали"),
                        ("jump", "Он перепрыгивает фигуры"),
                        ("double", "Он делает два хода подряд"),
                    ),
                    "correct_option": "combined",
                },
                "explanation": "Большой радиус ферзя связан с объединением возможностей ладьи и слона.",
                "sort_order": 1,
            }
        ],
    },
]


KING_LESSONS = [
    {
        "slug": "king-movement",
        "title": "Как ходит король",
        "theory": (
            "Король обычно перемещается на одну клетку в любом направлении: по вертикали, горизонтали или диагонали. "
            "При этом он не может перейти на поле, атакованное фигурой противника."
        ),
        "sort_order": 1,
        "tasks": [
            {
                "task_type": "select_squares",
                "title": "Король в центре",
                "instruction": "Отметьте все легальные клетки для белого короля e4.",
                "fen": "7k/8/8/8/4K3/8/8/8 w - - 0 1",
                "source_square": "e4",
                "difficulty": 1,
                "payload": {"piece": "K", "mode": "all_legal_moves"},
                "explanation": "Без угроз вокруг король e4 может перейти на восемь соседних клеток.",
                "sort_order": 1,
            },
            {
                "task_type": "make_move",
                "title": "Сделайте ход королём",
                "instruction": "Сделайте любой допустимый ход белым королём с e4.",
                "fen": "7k/8/8/8/4K3/8/8/8 w - - 0 1",
                "source_square": "e4",
                "difficulty": 1,
                "payload": {"piece": "K", "mode": "any_legal_move"},
                "explanation": "Король перемещается на одну соседнюю безопасную клетку.",
                "sort_order": 2,
            },
        ],
    },
    {
        "slug": "king-safety",
        "title": "Атакованные поля",
        "theory": (
            "Король не имеет права оставаться под шахом или самостоятельно переходить под удар. "
            "Поэтому часть геометрически соседних клеток может быть недоступна."
        ),
        "sort_order": 2,
        "tasks": [
            {
                "task_type": "select_squares",
                "title": "Уход с линии ладьи",
                "instruction": "Белому королю e4 объявлен шах ладьёй e8. Отметьте клетки, куда король может уйти.",
                "fen": "4r2k/8/8/8/4K3/8/8/8 w - - 0 1",
                "source_square": "e4",
                "difficulty": 2,
                "payload": {"piece": "K", "mode": "all_legal_moves"},
                "explanation": "Король должен покинуть вертикаль e и перейти на безопасное соседнее поле.",
                "sort_order": 1,
            },
            {
                "task_type": "choose_option",
                "title": "Можно ли войти под удар?",
                "instruction": "Разрешено ли королю сделать ход на поле, которое атакует фигура противника?",
                "fen": "7k/8/8/8/4K3/8/8/8 w - - 0 1",
                "source_square": "e4",
                "difficulty": 1,
                "payload": {
                    "options": _options(("no", "Нет"), ("yes", "Да")),
                    "correct_option": "no",
                },
                "explanation": "Нет. Король не может сделать ход, после которого окажется под шахом.",
                "sort_order": 2,
            },
        ],
    },
    {
        "slug": "king-capture",
        "title": "Взятие королём",
        "theory": (
            "Король может брать фигуры противника на соседних клетках, но только если конечная клетка безопасна."
        ),
        "sort_order": 3,
        "tasks": [
            {
                "task_type": "select_squares",
                "title": "Какие фигуры можно взять?",
                "instruction": "Отметьте клетки, на которых белый король e4 может взять фигуру противника.",
                "fen": "7k/8/8/3p1p2/4K3/8/8/8 w - - 0 1",
                "source_square": "e4",
                "difficulty": 2,
                "payload": {"piece": "K", "mode": "capture_squares"},
                "explanation": "Король может взять фигуру на d5 или f5, если после взятия эти поля безопасны.",
                "sort_order": 1,
            },
            {
                "task_type": "make_move",
                "title": "Выполните взятие королём",
                "instruction": "Сделайте белым королём e4 любое доступное взятие.",
                "fen": "7k/8/8/3p1p2/4K3/8/8/8 w - - 0 1",
                "source_square": "e4",
                "difficulty": 2,
                "payload": {"piece": "K", "mode": "legal_capture"},
                "explanation": "Король берёт так же, как ходит: на соседнее безопасное поле.",
                "sort_order": 2,
            },
        ],
    },
    {
        "slug": "king-opposition",
        "title": "Короли не стоят рядом",
        "theory": (
            "Два короля не могут находиться на соседних клетках: каждый король атакует все соседние поля."
        ),
        "sort_order": 4,
        "tasks": [
            {
                "task_type": "select_squares",
                "title": "Короли напротив друг друга",
                "instruction": "Отметьте все легальные клетки для белого короля e4.",
                "fen": "8/8/4k3/8/4K3/8/8/8 w - - 0 1",
                "source_square": "e4",
                "difficulty": 2,
                "payload": {"piece": "K", "mode": "all_legal_moves"},
                "explanation": "Белый король не может приближаться на соседнее поле к чёрному королю e6.",
                "sort_order": 1,
            }
        ],
    },
]


SPECIAL_RULES_LESSONS = [
    {
        "slug": "castling",
        "title": "Рокировка",
        "theory": (
            "Рокировка — особый ход короля и ладьи. Король перемещается на две клетки к ладье, а ладья становится рядом с ним. "
            "Рокировка разрешена только если король и соответствующая ладья не ходили, между ними нет фигур, король не под шахом "
            "и не проходит через атакованное поле."
        ),
        "sort_order": 1,
        "tasks": [
            {
                "task_type": "make_move",
                "title": "Выполните рокировку",
                "instruction": "Сделайте белыми любую разрешённую рокировку.",
                "fen": "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
                "source_square": "e1",
                "difficulty": 2,
                "payload": {
                    "piece": "K",
                    "mode": "accepted_moves",
                    "accepted_moves": ["e1g1", "e1c1"],
                },
                "explanation": "В этой позиции доступны короткая рокировка e1-g1 и длинная рокировка e1-c1.",
                "sort_order": 1,
            },
            {
                "task_type": "choose_option",
                "title": "Когда рокировка запрещена?",
                "instruction": "Можно ли рокироваться, если король в данный момент находится под шахом?",
                "fen": "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
                "source_square": "e1",
                "difficulty": 1,
                "payload": {
                    "options": _options(("no", "Нет"), ("yes", "Да")),
                    "correct_option": "no",
                },
                "explanation": "Нет. Из-под шаха рокироваться нельзя.",
                "sort_order": 2,
            },
        ],
    },
    {
        "slug": "en-passant",
        "title": "Взятие на проходе",
        "theory": (
            "Если пешка соперника с начальной клетки проходит сразу две клетки и оказывается рядом с вашей пешкой, "
            "её можно взять на проходе так, будто она прошла только одну клетку. Такое право существует только немедленно следующим ходом."
        ),
        "sort_order": 2,
        "tasks": [
            {
                "task_type": "make_move",
                "title": "Возьмите на проходе",
                "instruction": "Выполните белой пешкой e5 взятие на проходе.",
                "fen": "7k/8/8/3pP3/8/8/8/K7 w - d6 0 1",
                "source_square": "e5",
                "difficulty": 3,
                "payload": {"piece": "P", "mode": "accepted_moves", "accepted_moves": ["e5d6"]},
                "explanation": "Ход e5xd6 e.p. снимает чёрную пешку с d5, хотя белая пешка приходит на d6.",
                "sort_order": 1,
            },
            {
                "task_type": "choose_option",
                "title": "Когда действует право?",
                "instruction": "Как долго сохраняется возможность взятия на проходе?",
                "fen": "7k/8/8/3pP3/8/8/8/K7 w - d6 0 1",
                "source_square": "e5",
                "difficulty": 2,
                "payload": {
                    "options": _options(
                        ("immediate", "Только на следующем ходу"),
                        ("forever", "До конца партии"),
                        ("three", "В течение трёх ходов"),
                    ),
                    "correct_option": "immediate",
                },
                "explanation": "Взятие на проходе возможно только немедленно после двойного хода пешки соперника.",
                "sort_order": 2,
            },
        ],
    },
    {
        "slug": "promotion",
        "title": "Превращение пешки",
        "theory": (
            "Дойдя до последней горизонтали, пешка обязана превратиться в ферзя, ладью, слона или коня. "
            "Выбор новой фигуры не зависит от того, есть ли такая фигура уже на доске."
        ),
        "sort_order": 3,
        "tasks": [
            {
                "task_type": "select_squares",
                "title": "Поле превращения",
                "instruction": "Отметьте клетку, на которую должна пойти белая пешка e7 для превращения.",
                "fen": "7k/4P3/8/8/8/8/8/K7 w - - 0 1",
                "source_square": "e7",
                "difficulty": 2,
                "payload": {"piece": "P", "mode": "all_legal_moves"},
                "explanation": "Пешка e7 достигает последней горизонтали на e8 и после хода превращается в выбранную фигуру.",
                "sort_order": 1,
            },
            {
                "task_type": "choose_option",
                "title": "Во что можно превратить пешку?",
                "instruction": "Какой набор фигур разрешён при превращении пешки?",
                "fen": "7k/4P3/8/8/8/8/8/K7 w - - 0 1",
                "source_square": "e7",
                "difficulty": 1,
                "payload": {
                    "options": _options(
                        ("valid", "Ферзь, ладья, слон или конь"),
                        ("king", "Только ферзь или король"),
                        ("any", "Любая фигура, включая короля"),
                    ),
                    "correct_option": "valid",
                },
                "explanation": "Пешка может превратиться только в ферзя, ладью, слона или коня.",
                "sort_order": 2,
            },
        ],
    },
]


CHECK_MATE_STALEMATE_LESSONS = [
    {
        "slug": "check",
        "title": "Шах",
        "theory": (
            "Шах возникает, когда король атакован фигурой соперника. Игрок обязан устранить шах текущим ходом: "
            "уйти королём, закрыться другой фигурой или взять атакующую фигуру, если это возможно."
        ),
        "sort_order": 1,
        "tasks": [
            {
                "task_type": "make_move",
                "title": "Объявите шах ладьёй",
                "instruction": "Сделайте ход ладьёй a1, который объявляет шах чёрному королю.",
                "fen": "7k/8/8/8/8/8/8/R6K w - - 0 1",
                "source_square": "a1",
                "difficulty": 2,
                "payload": {"piece": "R", "mode": "accepted_moves", "accepted_moves": ["a1a8"]},
                "explanation": "После Ra8+ ладья атакует короля h8 по восьмой горизонтали.",
                "sort_order": 1,
            },
            {
                "task_type": "choose_option",
                "title": "Что делать при шахе?",
                "instruction": "Можно ли проигнорировать шах и сделать произвольный ход другой фигурой?",
                "fen": "4r2k/8/8/8/4K3/8/8/8 w - - 0 1",
                "source_square": "e4",
                "difficulty": 1,
                "payload": {
                    "options": _options(("no", "Нет, шах обязательно нужно устранить"), ("yes", "Да")),
                    "correct_option": "no",
                },
                "explanation": "Каждый легальный ответ на шах обязан вывести короля из-под атаки.",
                "sort_order": 2,
            },
        ],
    },
    {
        "slug": "escape-check",
        "title": "Защита от шаха",
        "theory": (
            "От шаха можно защититься тремя основными способами: уйти королём, взять атакующую фигуру или перекрыть линию атаки, "
            "если атакует дальнобойная фигура и между ней и королём есть место для блока."
        ),
        "sort_order": 2,
        "tasks": [
            {
                "task_type": "select_squares",
                "title": "Уход короля от шаха",
                "instruction": "Король e1 атакован ладьёй e8. Отметьте все клетки, куда король может уйти.",
                "fen": "4r2k/8/8/8/8/8/8/4K3 w - - 0 1",
                "source_square": "e1",
                "difficulty": 2,
                "payload": {"piece": "K", "mode": "all_legal_moves"},
                "explanation": "Король должен уйти с вертикали e на одну из безопасных соседних клеток.",
                "sort_order": 1,
            },
            {
                "task_type": "make_move",
                "title": "Уйдите из-под шаха",
                "instruction": "Сделайте белым королём e1 любой легальный ход, устраняющий шах.",
                "fen": "4r2k/8/8/8/8/8/8/4K3 w - - 0 1",
                "source_square": "e1",
                "difficulty": 2,
                "payload": {"piece": "K", "mode": "any_legal_move"},
                "explanation": "Легальный ход короля обязан вывести его из-под атаки ладьи.",
                "sort_order": 2,
            },
        ],
    },
    {
        "slug": "checkmate",
        "title": "Мат",
        "theory": (
            "Мат — это шах, от которого нет ни одного легального способа защиты. После мата партия немедленно заканчивается победой атакующей стороны."
        ),
        "sort_order": 3,
        "tasks": [
            {
                "task_type": "choose_option",
                "title": "Определите позицию",
                "instruction": "Что произошло с чёрным королём в показанной позиции?",
                "fen": "7k/6Q1/5K2/8/8/8/8/8 b - - 0 1",
                "source_square": None,
                "difficulty": 2,
                "payload": {
                    "options": _options(("mate", "Мат"), ("check", "Только шах"), ("stalemate", "Пат")),
                    "correct_option": "mate",
                },
                "explanation": "Чёрный король находится под шахом и не имеет ни одного легального ответа — это мат.",
                "sort_order": 1,
            },
            {
                "task_type": "make_move",
                "title": "Мат в один ход",
                "instruction": "Сделайте ферзём g6 ход, который ставит мат чёрному королю.",
                "fen": "7k/8/5KQ1/8/8/8/8/8 w - - 0 1",
                "source_square": "g6",
                "difficulty": 3,
                "payload": {"piece": "Q", "mode": "accepted_moves", "accepted_moves": ["g6g7"]},
                "explanation": "Qg7# атакует короля h8, а все клетки отступления перекрыты ферзём и белым королём.",
                "sort_order": 2,
            },
        ],
    },
    {
        "slug": "stalemate",
        "title": "Пат",
        "theory": (
            "Пат возникает, когда сторона должна ходить, её король не находится под шахом, но ни одного легального хода нет. "
            "Пат означает ничью, а не победу атакующей стороны."
        ),
        "sort_order": 4,
        "tasks": [
            {
                "task_type": "choose_option",
                "title": "Мат или пат?",
                "instruction": "Определите состояние чёрных в показанной позиции.",
                "fen": "7k/5K2/6Q1/8/8/8/8/8 b - - 0 1",
                "source_square": None,
                "difficulty": 2,
                "payload": {
                    "options": _options(("stalemate", "Пат"), ("mate", "Мат"), ("normal", "Позиция продолжается")),
                    "correct_option": "stalemate",
                },
                "explanation": "Король h8 не атакован, но у чёрных нет легального хода — это пат и ничья.",
                "sort_order": 1,
            },
            {
                "task_type": "make_move",
                "title": "Создайте пат",
                "instruction": "Сделайте ферзём g5 ход, после которого у чёрных возникнет пат.",
                "fen": "7k/5K2/8/6Q1/8/8/8/8 w - - 0 1",
                "source_square": "g5",
                "difficulty": 3,
                "payload": {"piece": "Q", "mode": "accepted_moves", "accepted_moves": ["g5g6"]},
                "explanation": "После Qg6 чёрный король не под шахом, но не имеет ни одного легального хода — возникает пат.",
                "sort_order": 2,
            },
        ],
    },
]


LESSONS_BY_MODULE = {
    "pawn": PAWN_LESSONS,
    "knight": KNIGHT_LESSONS,
    "bishop": BISHOP_LESSONS,
    "rook": ROOK_LESSONS,
    "queen": QUEEN_LESSONS,
    "king": KING_LESSONS,
    "special-rules": SPECIAL_RULES_LESSONS,
    "check-mate-stalemate": CHECK_MATE_STALEMATE_LESSONS,
}
