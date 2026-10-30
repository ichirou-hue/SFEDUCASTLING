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
        "slug": "pieces",
        "title": "Фигуры: ходы и взятия",
        "description": "Как ходят и берут фигуры: от пешки до короля.",
        "sort_order": 1,
        "enabled": True,
    },
    {
        "slug": "king-safety",
        "title": "Шах, мат и безопасность короля",
        "description": "Шах и защита от него, мат, пат и безопасность короля - включая рокировку.",
        "sort_order": 2,
        "enabled": True,
    },
    {
        "slug": "tactics",
        "title": "Тактические приёмы",
        "description": "Незащищённые фигуры, двойной удар, вилка, связка, уничтожение защитника, открытое нападение, вскрытый шах и комбинации.",
        "sort_order": 3,
        "enabled": True,
    },
    {
        "slug": "endgames",
        "title": "Эндшпиль",
        "description": "Мат тяжёлыми фигурами и двумя ладьями, проведение проходной пешки, пешечный прорыв, правило квадрата и оппозиция королей - а также взятие на проходе и превращение.",
        "sort_order": 4,
        "enabled": True,
    },
    {
        "slug": "openings",
        "title": "Дебюты",
        "description": "Первые ходы партии: классические начала, борьба за центр и развитие фигур.",
        "sort_order": 5,
        "enabled": True,
    },
    {
        "slug": "middlegame",
        "title": "Миттельшпиль: стратегия",
        "description": "Материальное и позиционное преимущество, расположение пешек, открытые линии, хорошие и плохие фигуры и типовые ошибки.",
        "sort_order": 6,
        "enabled": True,
    },
]



def _options(*pairs: tuple[str, str]) -> list[dict[str, str]]:
    return [{"id": option_id, "label": label} for option_id, label in pairs]


PIECES_LESSONS = [
    {
        "slug": "pawn-basic-movement",
        "title": "Обычный ход пешки",
        "section_title": "Пешка",
        "theory": (
            "Пешка ходит только вперёд по своей вертикали. Обычно она перемещается на одну клетку. "
            "Белые пешки движутся в сторону восьмой горизонтали, чёрные - в сторону первой."
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
        "section_title": "Пешка",
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
                "fen": "7k/1nB1pp2/p6b/8/1pP1P1PP/3r4/4NP2/K7 w - - 0 24",
                "source_square": "e4",
                "difficulty": 1,
                "payload": {"piece": "P", "mode": "all_legal_moves"},
                "explanation": "Пешка с e4 уже не может идти на две клетки: её обычный ход - e4-e5.",
                "sort_order": 1,
            },
            {
                "task_type": "choose_option",
                "title": "Направление движения",
                "instruction": "Как белая пешка ходит без взятия?",
                "fen": "7k/1nB1pp2/p6b/8/1pP1P1PP/3r4/4NP2/K7 w - - 0 24",
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
        "section_title": "Пешка",
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
                "fen": "5r1k/2Bn1p2/2N5/1ppp1p2/p3P3/P3P2b/1P1P4/K2R4 w - - 0 31",
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
                "fen": "5r1k/2Bn1p2/2N5/1ppp1p2/p3P3/P3P2b/1P1P4/K2R4 w - - 0 31",
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
        "section_title": "Пешка",
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
                "fen": "3N3k/p7/1p6/2pppP2/3P3p/n6r/P2bPPR1/K2B4 w - - 0 28",
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
                "fen": "7k/n1pN4/6bp/P2p4/P2P1P1B/1R6/4Q3/K7 w - - 0 38",
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
        "section_title": "Пешка",
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
                "fen": "7k/3n1p2/8/P1p1pr1p/3Ppp1B/N2b4/P4PP1/K5QR w - - 0 29",
                "source_square": "d4",
                "difficulty": 2,
                "payload": {"piece": "P", "mode": "all_legal_moves"},
                "explanation": "Пешка может пойти на d5 либо взять фигуру на c5 или e5.",
                "sort_order": 1,
            }
        ],
    },
    {
        "slug": "knight-movement",
        "title": "Как ходит конь",
        "section_title": "Конь",
        "theory": (
            "Конь перемещается буквой «Г»: на две клетки по вертикали или горизонтали "
            "и затем на одну клетку перпендикулярно. Из центра доски у коня может быть "
            "до восьми вариантов хода."
        ),
        "sort_order": 6,
        "tasks": [
            {
                "task_type": "select_squares",
                "title": "Конь в центре доски",
                "instruction": "Отметьте все клетки, на которые может перейти белый конь с e4.",
                "fen": "3r3k/1nq2p1p/8/pP2p2N/4N3/1R5P/1P4Pb/K4B2 w - - 0 40",
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
        "section_title": "Конь",
        "theory": (
            "Чем ближе конь к краю или углу доски, тем меньше клеток ему доступно. "
            "Поэтому конь обычно активнее ближе к центру."
        ),
        "sort_order": 7,
        "tasks": [
            {
                "task_type": "select_squares",
                "title": "Конь в углу a1",
                "instruction": "Отметьте все клетки, доступные белому коню с a1.",
                "fen": "3N3k/4R3/r6n/pb5p/2q3pp/P7/3P1PPB/N3K2Q w - - 0 39",
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
                "fen": "k2B3N/pq6/p7/P3PR1P/1nP1p1pN/7b/1r6/4K3 w - - 0 35",
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
        "section_title": "Конь",
        "theory": (
            "Конь - единственная шахматная фигура, которая может перепрыгивать через другие фигуры. "
            "Фигуры рядом с ним не блокируют ход, но конечная клетка не может быть занята своей фигурой."
        ),
        "sort_order": 8,
        "tasks": [
            {
                "task_type": "select_squares",
                "title": "Окружённый конь",
                "instruction": "Конь d4 окружён своими пешками. Отметьте все клетки, на которые он всё равно может перейти.",
                "fen": "1q1b3k/n1p1p2r/8/p1PPP3/p1PNP1B1/2PPP1R1/5N2/K4Q2 w - - 0 19",
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
                "fen": "1q1b3k/n1p1p2r/8/p1PPP3/p1PNP1B1/2PPP1R1/5N2/K4Q2 w - - 0 19",
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
        "section_title": "Конь",
        "theory": (
            "Конь берёт фигуру противника на той клетке, на которую приходит своим обычным ходом буквой «Г». "
            "Перепрыгиваемые фигуры при этом не снимаются с доски."
        ),
        "sort_order": 9,
        "tasks": [
            {
                "task_type": "select_squares",
                "title": "Найдите доступные взятия",
                "instruction": "Отметьте клетки, на которых белый конь e4 может взять фигуру противника.",
                "fen": "7k/5p2/1r6/ppr1P1bp/4N2N/P5BR/4b1PP/K2n4 w - - 0 40",
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
                "fen": "7k/5p2/1r6/ppr1P1bp/4N2N/P5BR/4b1PP/K2n4 w - - 0 40",
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
        "section_title": "Конь",
        "theory": (
            "В итоговых заданиях вспомните форму хода коня, влияние края доски, "
            "возможность перепрыгивания и правила взятия."
        ),
        "sort_order": 10,
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
    {
        "slug": "bishop-movement",
        "title": "Как ходит слон",
        "section_title": "Слон",
        "theory": (
            "Слон перемещается на любое число свободных клеток по диагонали. Он не может менять цвет полей: "
            "слон, начавший на светлом поле, всю партию остаётся на светлых полях."
        ),
        "sort_order": 11,
        "tasks": [
            {
                "task_type": "select_squares",
                "title": "Слон в центре",
                "instruction": "Отметьте все клетки, на которые может перейти белый слон d4.",
                "fen": "K7/3np2k/4p3/PP4pB/p2BP3/6b1/N2P4/5r2 w - - 0 24",
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
                "fen": "K7/3np2k/4p3/PP4pB/p2BP3/6b1/N2P4/5r2 w - - 0 24",
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
        "section_title": "Слон",
        "theory": (
            "Слон не умеет перепрыгивать фигуры. Своя фигура останавливает его движение, а фигуру противника "
            "можно взять, но двигаться дальше за неё нельзя."
        ),
        "sort_order": 12,
        "tasks": [
            {
                "task_type": "select_squares",
                "title": "Слон между своими пешками",
                "instruction": "Отметьте все легальные клетки для слона d4.",
                "fen": "K6k/4p1rp/p5qp/P1P1P1b1/1P1B4/1P3Q2/4R1PN/3n3B w - - 0 33",
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
                "fen": "K2N4/7k/p7/2Pp2PB/p2B1p2/n2P3b/P6P/4Q3 w - - 0 20",
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
        "section_title": "Слон",
        "theory": (
            "Слон берёт фигуру противника, если она находится на его диагонали и между ними нет других фигур."
        ),
        "sort_order": 13,
        "tasks": [
            {
                "task_type": "select_squares",
                "title": "Две цели на диагоналях",
                "instruction": "Отметьте клетки, на которых слон d4 может взять фигуру противника.",
                "fen": "3N2nk/q3p2p/1r3nb1/pP5p/3B4/6P1/1P4P1/K1Q2BR1 w - - 0 36",
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
                "fen": "3N2nk/q3p2p/1r3nb1/pP5p/3B4/6P1/1P4P1/K1Q2BR1 w - - 0 36",
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
        "section_title": "Слон",
        "theory": "Слон всегда движется по диагонали, не перепрыгивает фигуры и сохраняет цвет своих полей.",
        "sort_order": 14,
        "tasks": [
            {
                "task_type": "choose_option",
                "title": "Цвет полей слона",
                "instruction": "Может ли слон в течение партии перейти с белых полей на чёрные?",
                "fen": "K7/3np2k/4p3/PP4pB/p2BP3/6b1/N2P4/5r2 w - - 0 24",
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
    {
        "slug": "rook-movement",
        "title": "Как ходит ладья",
        "section_title": "Ладья",
        "theory": "Ладья ходит на любое число свободных клеток по вертикали или горизонтали.",
        "sort_order": 15,
        "tasks": [
            {
                "task_type": "select_squares",
                "title": "Ладья в центре",
                "instruction": "Отметьте все клетки, на которые может перейти белая ладья d4.",
                "fen": "K6k/2p5/b1p2n2/p4Bp1/3R4/PP3PP1/5N2/4R3 w - - 0 31",
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
                "fen": "K6k/2p5/b1p2n2/p4Bp1/3R4/PP3PP1/5N2/4R3 w - - 0 31",
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
        "section_title": "Ладья",
        "theory": "Ладья не перепрыгивает через фигуры. Свои фигуры ограничивают её линию движения.",
        "sort_order": 16,
        "tasks": [
            {
                "task_type": "select_squares",
                "title": "Ладья и свои фигуры",
                "instruction": "Отметьте все легальные клетки для ладьи d4.",
                "fen": "K6k/2R1N3/3Ppp1n/1P5p/3R1PpP/Pb2r3/7P/5B2 w - - 0 18",
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
                "fen": "K6k/5p2/NppP1p1b/5B1P/3R4/nP3P2/4R3/6r1 w - - 0 40",
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
        "section_title": "Ладья",
        "theory": "Ладья может взять первую фигуру противника на своей вертикали или горизонтали.",
        "sort_order": 17,
        "tasks": [
            {
                "task_type": "select_squares",
                "title": "Цели ладьи",
                "instruction": "Отметьте клетки, на которых ладья d4 может взять фигуру противника.",
                "fen": "3b3k/7r/p2r2p1/6p1/1P1R2bp/P1N5/3PRPBn/K7 w - - 0 23",
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
                "fen": "3b3k/7r/p2r2p1/6p1/1P1R2bp/P1N5/3PRPBn/K7 w - - 0 23",
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
        "section_title": "Ладья",
        "theory": "Ладья сильнее всего действует по открытым вертикалям и горизонталям, где ей не мешают другие фигуры.",
        "sort_order": 18,
        "tasks": [
            {
                "task_type": "choose_option",
                "title": "Направления ладьи",
                "instruction": "Какие направления доступны ладье?",
                "fen": "K6k/2p5/b1p2n2/p4Bp1/3R4/PP3PP1/5N2/4R3 w - - 0 31",
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
    {
        "slug": "queen-movement",
        "title": "Как ходит ферзь",
        "section_title": "Ферзь",
        "theory": (
            "Ферзь объединяет движения ладьи и слона: он может идти на любое число свободных клеток "
            "по вертикали, горизонтали или диагонали."
        ),
        "sort_order": 19,
        "tasks": [
            {
                "task_type": "select_squares",
                "title": "Ферзь в центре",
                "instruction": "Отметьте все клетки, на которые может перейти белый ферзь d4.",
                "fen": "K1N5/7k/8/pb4p1/3Q4/PP3P2/n6P/4R3 w - - 0 18",
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
                "fen": "K1N5/7k/8/pb4p1/3Q4/PP3P2/n6P/4R3 w - - 0 18",
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
        "section_title": "Ферзь",
        "theory": "Несмотря на большой радиус действия, ферзь, как ладья и слон, не умеет перепрыгивать фигуры.",
        "sort_order": 20,
        "tasks": [
            {
                "task_type": "select_squares",
                "title": "Ферзь среди своих фигур",
                "instruction": "Отметьте все легальные клетки для ферзя d4.",
                "fen": "4b1rk/5qp1/p2P4/N1P1Ppp1/3Q4/n1P1P3/1R4PB/K7 w - - 0 35",
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
                "fen": "K1N5/7k/8/pb4p1/3Q4/PP3P2/n6P/4R3 w - - 0 18",
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
        "section_title": "Ферзь",
        "theory": "Ферзь берёт первую фигуру противника на доступной вертикали, горизонтали или диагонали.",
        "sort_order": 21,
        "tasks": [
            {
                "task_type": "select_squares",
                "title": "Цели ферзя",
                "instruction": "Отметьте клетки, на которых ферзь d4 может взять фигуру противника.",
                "fen": "6qk/p3p2n/1r1p1nr1/bp3p2/3Q2bP/P5P1/6PR/KN5B w - - 0 29",
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
                "fen": "6qk/p3p2n/1r1p1nr1/bp3p2/3Q2bP/P5P1/6PR/KN5B w - - 0 29",
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
        "section_title": "Ферзь",
        "theory": "Ферзь очень подвижен, но подчиняется тем же ограничениям линий, что ладья и слон.",
        "sort_order": 22,
        "tasks": [
            {
                "task_type": "choose_option",
                "title": "Сильная сторона ферзя",
                "instruction": "Почему ферзь контролирует так много клеток?",
                "fen": "K1N5/7k/8/pb4p1/3Q4/PP3P2/n6P/4R3 w - - 0 18",
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
    {
        "slug": "king-movement",
        "title": "Как ходит король",
        "section_title": "Король",
        "theory": (
            "Король обычно перемещается на одну клетку в любом направлении: по вертикали, горизонтали или диагонали. "
            "При этом он не может перейти на поле, атакованное фигурой противника."
        ),
        "sort_order": 23,
        "tasks": [
            {
                "task_type": "select_squares",
                "title": "Король в центре",
                "instruction": "Отметьте все легальные клетки для белого короля e4.",
                "fen": "2n1bN1k/1pp5/7p/2P5/4K2p/PPP5/6r1/3R4 w - - 0 34",
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
                "fen": "2n1bN1k/1pp5/7p/2P5/4K2p/PPP5/6r1/3R4 w - - 0 34",
                "source_square": "e4",
                "difficulty": 1,
                "payload": {"piece": "K", "mode": "any_legal_move"},
                "explanation": "Король перемещается на одну соседнюю безопасную клетку.",
                "sort_order": 2,
            },
        ],
    },
    {
        "slug": "king-capture",
        "title": "Взятие королём",
        "section_title": "Король",
        "theory": (
            "Король может брать фигуры противника на соседних клетках, но только если конечная клетка безопасна."
        ),
        "sort_order": 24,
        "tasks": [
            {
                "task_type": "select_squares",
                "title": "Какие фигуры можно взять?",
                "instruction": "Отметьте клетки, на которых белый король e4 может взять фигуру противника.",
                "fen": "3b3k/2rpp3/B7/P1pnPnp1/4K3/1R6/1P2P3/7N w - - 0 33",
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
                "fen": "3b3k/2rpp3/B7/P1pnPnp1/4K3/1R6/1P2P3/7N w - - 0 33",
                "source_square": "e4",
                "difficulty": 2,
                "payload": {"piece": "K", "mode": "legal_capture"},
                "explanation": "Король берёт так же, как ходит: на соседнее безопасное поле.",
                "sort_order": 2,
            },
        ],
    },
    {
        "slug": "king-review",
        "title": "Закрепление: король",
        "section_title": "Король",
        "theory": (
            "Король ходит на одну клетку в любом направлении, но не может встать под шах. "
            "В центре доски у него до восьми полей, у края - меньше. Король не может также "
            "приблизиться вплотную к вражескому королю: между ними всегда должно оставаться "
            "хотя бы одно поле."
        ),
        "sort_order": 25,
        "tasks": [
            {
                "task_type": "select_squares",
                "title": "Пять полей короля",
                "instruction": "Отметьте все клетки, на которые может пойти белый король с e1.",
                "fen": "2r1k1nB/5q2/4p3/1p1p2P1/3P1p2/QP6/6PR/1b2K2N w - - 0 25",
                "source_square": "e1",
                "difficulty": 1,
                "payload": {"piece": "K", "mode": "all_legal_moves"},
                "explanation": "С e1 король может пойти на d1, f1, d2, e2 и f2 - всего пять полей.",
                "sort_order": 1,
            },
            {
                "task_type": "choose_option",
                "title": "Атакованное поле",
                "instruction": "Чёрная ладья стоит на f8. На какое из перечисленных полей белому королю с e1 идти нельзя?",
                "fen": "5r1k/1pb1p3/4B3/7p/1P1pP2P/r7/1nP5/4K2N w - - 0 30",
                "source_square": "e1",
                "difficulty": 2,
                "payload": {
                    "options": _options(
                        ("d1", "d1"),
                        ("f1", "f1"),
                        ("e2", "e2"),
                        ("d2", "d2"),
                    ),
                    "correct_option": "f1",
                },
                "explanation": "Ладья f8 простреливает вертикаль f, поэтому на f1 король оказался бы под шахом.",
                "sort_order": 2,
            },
        ],
    },
]


KING_SAFETY_LESSONS = [
    {
        "slug": "check",
        "title": "Шах",
        "section_title": "Шах и защита",
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
                "fen": "7k/1p1qr3/6R1/1p5N/1P1ppB2/1P1b2P1/1n1P4/R6K w - - 0 19",
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
                "fen": "4r2k/8/3p1p1p/7P/P1B1K2p/P1P5/1R6/n3N3 w - - 0 33",
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
        "section_title": "Шах и защита",
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
                "fen": "3rr2k/n2p4/5R2/1pP2P2/1p1P2pN/6P1/2Q5/B3K2b w - - 0 28",
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
                "fen": "3rr2k/n2p4/5R2/1pP2P2/1p1P2pN/6P1/2Q5/B3K2b w - - 0 28",
                "source_square": "e1",
                "difficulty": 2,
                "payload": {"piece": "K", "mode": "any_legal_move"},
                "explanation": "Легальный ход короля обязан вывести его из-под атаки ладьи.",
                "sort_order": 2,
            },
        ],
    },
    {
        "slug": "king-safety",
        "title": "Атакованные поля",
        "section_title": "Шах и защита",
        "theory": (
            "Король не имеет права оставаться под шахом или самостоятельно переходить под удар. "
            "Поэтому часть геометрически соседних клеток может быть недоступна."
        ),
        "sort_order": 3,
        "tasks": [
            {
                "task_type": "select_squares",
                "title": "Уход с линии ладьи",
                "instruction": "Белому королю e4 объявлен шах ладьёй e8. Отметьте клетки, куда король может уйти.",
                "fen": "4r2k/8/3p1p1p/7P/P1B1K2p/P1P5/1R6/n3N3 w - - 0 33",
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
                "fen": "2n1bN1k/1pp5/7p/2P5/4K2p/PPP5/6r1/3R4 w - - 0 34",
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
        "slug": "checkmate",
        "title": "Мат",
        "section_title": "Мат и пат",
        "theory": (
            "Мат - это шах, от которого нет ни одного легального способа защиты. После мата партия немедленно заканчивается победой атакующей стороны."
        ),
        "sort_order": 4,
        "tasks": [
            {
                "task_type": "choose_option",
                "title": "Определите позицию",
                "instruction": "Что произошло с чёрным королём в показанной позиции?",
                "fen": "7k/6Q1/p2p1K2/B6p/P2NP2p/P2n3b/7P/1r6 b - - 0 22",
                "source_square": None,
                "difficulty": 2,
                "payload": {
                    "options": _options(("mate", "Мат"), ("check", "Только шах"), ("stalemate", "Пат")),
                    "correct_option": "mate",
                },
                "explanation": "Чёрный король находится под шахом и не имеет ни одного легального ответа - это мат.",
                "sort_order": 1,
            },
            {
                "task_type": "make_move",
                "title": "Мат в один ход",
                "instruction": "Сделайте ферзём g6 ход, который ставит мат чёрному королю.",
                "fen": "7k/3p4/3b1KQ1/pP1Pp3/np5r/4B3/3P1P2/1NR5 w - - 0 31",
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
        "section_title": "Мат и пат",
        "theory": (
            "Пат возникает, когда сторона должна ходить, её король не находится под шахом, но ни одного легального хода нет. "
            "Пат означает ничью, а не победу атакующей стороны."
        ),
        "sort_order": 5,
        "tasks": [
            {
                "task_type": "choose_option",
                "title": "Мат или пат?",
                "instruction": "Определите состояние чёрных в показанной позиции.",
                "fen": "7k/5K2/6Q1/1PP5/8/R7/3PNP1B/8 b - - 0 37",
                "source_square": None,
                "difficulty": 2,
                "payload": {
                    "options": _options(("stalemate", "Пат"), ("mate", "Мат"), ("normal", "Позиция продолжается")),
                    "correct_option": "stalemate",
                },
                "explanation": "Король h8 не атакован, но у чёрных нет легального хода - это пат и ничья.",
                "sort_order": 1,
            },
            {
                "task_type": "make_move",
                "title": "Создайте пат",
                "instruction": "Сделайте ферзём g5 ход, после которого у чёрных возникнет пат.",
                "fen": "1r5k/3R1K1p/1p6/6Q1/p1p5/2P2B2/1PN1Pn1P/8 w - - 0 34",
                "source_square": "g5",
                "difficulty": 3,
                "payload": {"piece": "Q", "mode": "accepted_moves", "accepted_moves": ["g5g6"]},
                "explanation": "После Qg6 чёрный король не под шахом, но не имеет ни одного легального хода - возникает пат.",
                "sort_order": 2,
            },
        ],
    },
    {
        "slug": "king-opposition",
        "title": "Короли не стоят рядом",
        "section_title": "Безопасность короля",
        "theory": (
            "Два короля не могут находиться на соседних клетках: каждый король атакует все соседние поля."
        ),
        "sort_order": 6,
        "tasks": [
            {
                "task_type": "select_squares",
                "title": "Короли напротив друг друга",
                "instruction": "Отметьте все легальные клетки для белого короля e4.",
                "fen": "6R1/2p5/4k2B/n1P4p/pp2K2b/8/1P2P2P/7N w - - 0 40",
                "source_square": "e4",
                "difficulty": 2,
                "payload": {"piece": "K", "mode": "all_legal_moves"},
                "explanation": "Белый король не может приближаться на соседнее поле к чёрному королю e6.",
                "sort_order": 1,
            }
        ],
    },
    {
        "slug": "castling",
        "title": "Рокировка",
        "section_title": "Безопасность короля",
        "theory": (
            "Рокировка - особый ход короля и ладьи. Король перемещается на две клетки к ладье, а ладья становится рядом с ним. "
            "Рокировка разрешена только если король и соответствующая ладья не ходили, между ними нет фигур, король не под шахом "
            "и не проходит через атакованное поле."
        ),
        "sort_order": 7,
        "tasks": [
            {
                "task_type": "make_move",
                "title": "Выполните рокировку",
                "instruction": "Сделайте белыми любую разрешённую рокировку.",
                "fen": "r3k2r/6p1/2nB4/1p1p1pP1/8/1PP5/1N4P1/R3K2R w KQkq - 0 32",
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
                "fen": "r3k2r/6p1/2nB4/1p1p1pP1/8/1PP5/1N4P1/R3K2R w KQkq - 0 32",
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
        "slug": "back-rank-mate",
        "title": "Мат по последней горизонтали",
        "section_title": "Мат и пат",
        "theory": (
            "Даже защищённый король может попасть в мат, если его собственные пешки перекрывают "
            "поля отступления. Ладья или ферзь, врываясь на последнюю горизонталь, нередко ставят "
            "мат такому «запертому» королю. Опасность последней горизонтали - одна из самых частых "
            "причин поражения в партиях."
        ),
        "sort_order": 8,
        "tasks": [
            {
                "task_type": "make_move",
                "title": "Мат ладьёй",
                "instruction": "Чёрный король заперт за своими пешками. Поставьте мат ладьёй по последней горизонтали.",
                "fen": "1n1b2k1/1p3ppp/8/2r5/Q7/5PNP/1P1P4/4R1K1 w - - 0 19",
                "source_square": "e1",
                "difficulty": 2,
                "payload": {"piece": "R", "mode": "accepted_moves", "accepted_moves": ["e1e8"]},
                "explanation": "После Re8 ладья бьёт короля по горизонтали, а поля отступления перекрыты пешками f7, g7, h7.",
                "sort_order": 1,
            },
            {
                "task_type": "choose_option",
                "title": "Почему это мат",
                "instruction": "Почему ход ладьёй на e8 заканчивает партию матом?",
                "fen": "1n1b2k1/1p3ppp/8/2r5/Q7/5PNP/1P1P4/4R1K1 w - - 0 19",
                "source_square": "e1",
                "difficulty": 2,
                "payload": {
                    "options": _options(
                        ("trapped", "Король заперт своими пешками, а ладья бьёт по всей последней горизонтали"),
                        ("defended", "Ладья защищена королём"),
                        ("stalemate", "Это не мат, а пат"),
                        ("only_check", "Это шах без продолжения"),
                    ),
                    "correct_option": "trapped",
                },
                "explanation": "Свои пешки перекрывают королю поля g7, h7 и f7, а ладья держит всю восьмую горизонталь.",
                "sort_order": 2,
            },
        ],
    },
    {
        "slug": "smothered-mate",
        "title": "Спёртый мат",
        "section_title": "Мат и пат",
        "theory": (
            "Спёртый мат - это мат конём королю, который зажат своими же фигурами. Конь ставит шах "
            "с поля, недоступного для фигур атакуемого короля, и тому некуда уйти. Это один из самых "
            "красивых матовых приёмов."
        ),
        "sort_order": 9,
        "tasks": [
            {
                "task_type": "make_move",
                "title": "Мат конём",
                "instruction": "Король h8 окружён своими фигурами. Поставьте спёртый мат конём.",
                "fen": "6rk/bp2p1pp/8/P3N1Pp/Pn2Pp1N/2r5/3R4/K4B2 w - - 0 32",
                "source_square": "e5",
                "difficulty": 3,
                "payload": {"piece": "N", "mode": "accepted_moves", "accepted_moves": ["e5f7"]},
                "explanation": "Nf7 даёт шах. Король не может ни уйти (свои ладья g8 и пешки g7, h7), ни взять коня - это мат.",
                "sort_order": 1,
            },
            {
                "task_type": "choose_option",
                "title": "Что такое спёртый мат",
                "instruction": "В чём особенность спёртого мата?",
                "fen": "6rk/bp2p1pp/8/P3N1Pp/Pn2Pp1N/2r5/3R4/K4B2 w - - 0 32",
                "source_square": "e5",
                "difficulty": 3,
                "payload": {
                    "options": _options(
                        ("knight_trapped", "Мат конём королю, зажатому собственными фигурами"),
                        ("queen_mate", "Мат ферзём по последней горизонтали"),
                        ("double_rook", "Мат двумя ладьями"),
                        ("pawn_mate", "Мат пешкой на последней горизонтали"),
                    ),
                    "correct_option": "knight_trapped",
                },
                "explanation": "Ключ к спёртому мату - король не может двинуться, потому что все соседние поля заняты его же фигурами.",
                "sort_order": 2,
            },
        ],
    },
    {
        "slug": "double-check",
        "title": "Двойной шах",
        "section_title": "Шах и защита",
        "theory": (
            "Двойной шах - это шах сразу от двух фигур. От него нельзя защититься, закрыв линию или "
            "взяв атакующую фигуру: король обязан отступить. Чаще всего двойной шах возникает, когда "
            "фигура уходит с линии и открывает шах своей же дальней фигуре, продолжая бить сама."
        ),
        "sort_order": 10,
        "tasks": [
            {
                "task_type": "make_move",
                "title": "Откройте двойной шах",
                "instruction": "Сделайте ход конём, который одновременно открывает шах слону и бьёт сам.",
                "fen": "3N2k1/3p3p/3p4/3N1PPP/2BPp2p/r7/7n/1K6 w - - 0 27",
                "source_square": "d5",
                "difficulty": 3,
                "payload": {"piece": "N", "mode": "accepted_moves", "accepted_moves": ["d5f6"]},
                "explanation": "Конь уходит с d5 и с поля f6 бьёт короля сам, открывая диагональ слону c4 - это двойной шах.",
                "sort_order": 1,
            },
            {
                "task_type": "choose_option",
                "title": "Чем опасен двойной шах",
                "instruction": "Почему двойной шах особенно опасен?",
                "fen": "3N2k1/3p3p/3p4/3N1PPP/2BPp2p/r7/7n/1K6 w - - 0 27",
                "source_square": "d5",
                "difficulty": 3,
                "payload": {
                    "options": _options(
                        ("king_must_move", "От него нельзя закрыться или взять одну фигуру - король обязан уйти"),
                        ("extra_tempo", "Он даёт лишний темп на рокировку"),
                        ("draw", "Он всегда приводит к ничьей"),
                        ("blocked", "Его можно закрыть, как обычный шах"),
                    ),
                    "correct_option": "king_must_move",
                },
                "explanation": "Даже если убрать одну атакующую фигуру, шах остаётся от другой - поэтому спасает только ход королём.",
                "sort_order": 2,
            },
        ],
    },
]


OPENINGS_LESSONS = [
    {
        "slug": "italian",
        "title": "Итальянская партия",
        "section_title": "Открытые дебюты",
        "theory": (
            "Итальянская партия начинается ходами 1.e4 e5 2.Nf3 Nc6 3.Bc4. Белые развивают коня на f3, "
            "а затем выводят слона на c4, нацеливаясь на слабую пешку f7 перед чёрным королём. "
            "Чёрные отвечают 3...Bc5, повторяя угрозу на f2. После 4.c3 белые подготавливают ход d2-d4 "
            "и получают прочную пешечную цепь в центре."
        ),
        "sort_order": 1,
        "tasks": [
            {
                "task_type": "make_move",
                "title": "Первый ход итальянской партии",
                "instruction": "Белые открывают игру пешкой. Сделайте ход e2-e4.",
                "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                "source_square": "e2",
                "difficulty": 1,
                "payload": {"piece": "P", "mode": "accepted_moves", "accepted_moves": ["e2e4"]},
                "explanation": "1.e4 занимает центр и открывает диагонали слону f1 и ферзю d1.",
                "sort_order": 1,
            },
            {
                "task_type": "make_move",
                "title": "Развитие коня",
                "instruction": "Чёрные ответили 1...e5. Развивайте коня g1 на f3.",
                "fen": "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
                "source_square": "g1",
                "difficulty": 1,
                "payload": {"piece": "N", "mode": "accepted_moves", "accepted_moves": ["g1f3"]},
                "explanation": "2.Nf3 атакует пешку e5 и развивает коня к центру.",
                "sort_order": 2,
            },
            {
                "task_type": "make_move",
                "title": "Слон на c4",
                "instruction": "Чёрные сыграли 2...Nc6. Выведите слона f1 на c4.",
                "fen": "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
                "source_square": "f1",
                "difficulty": 2,
                "payload": {"piece": "B", "mode": "accepted_moves", "accepted_moves": ["f1c4"]},
                "explanation": "3.Bc4 готовит давление на пешку f7 - слабое место чёрных в начале партии.",
                "sort_order": 3,
            },
            {
                "task_type": "make_move",
                "title": "Подготовка d2-d4",
                "instruction": "После 3...Bc5 подготовьте продвижение пешки центра ходом c2-c3.",
                "fen": "r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
                "source_square": "c2",
                "difficulty": 3,
                "payload": {"piece": "P", "mode": "accepted_moves", "accepted_moves": ["c2c3"]},
                "explanation": "4.c3 поддерживает будущий ход d2-d4 и забирает у чёрных поле d4 для их коня.",
                "sort_order": 4,
            },
        ],
    },
    {
        "slug": "ruy-lopez",
        "title": "Испанская партия",
        "section_title": "Открытые дебюты",
        "theory": (
            "Испанская партия (дебют Руи Лопеса) возникает после 1.e4 e5 2.Nf3 Nc6 3.Bb5. "
            "Слон b5 не берёт коня, но давит на него, создавая угрозу пешке e5. Это один из самых "
            "гибких и популярных дебютов на всех уровнях игры."
        ),
        "sort_order": 2,
        "tasks": [
            {
                "task_type": "make_move",
                "title": "Начало испанской партии",
                "instruction": "Сделайте первый ход белыми - e2-e4.",
                "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                "source_square": "e2",
                "difficulty": 1,
                "payload": {"piece": "P", "mode": "accepted_moves", "accepted_moves": ["e2e4"]},
                "explanation": "1.e4 открывает партию и борется за центральные поля.",
                "sort_order": 1,
            },
            {
                "task_type": "make_move",
                "title": "Конь против пешки e5",
                "instruction": "После 1...e5 выведите коня g1 на f3.",
                "fen": "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
                "source_square": "g1",
                "difficulty": 1,
                "payload": {"piece": "N", "mode": "accepted_moves", "accepted_moves": ["g1f3"]},
                "explanation": "2.Nf3 атакует пешку e5 и развивает коня.",
                "sort_order": 2,
            },
            {
                "task_type": "make_move",
                "title": "Слон на b5",
                "instruction": "Чёрные защитили пешку ходом 2...Nc6. Поставьте слона f1 на b5.",
                "fen": "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
                "source_square": "f1",
                "difficulty": 2,
                "payload": {"piece": "B", "mode": "accepted_moves", "accepted_moves": ["f1b5"]},
                "explanation": "3.Bb5 давит на коня c6 и косвенно угрожает пешке e5.",
                "sort_order": 3,
            },
        ],
    },
    {
        "slug": "sicilian",
        "title": "Сицилианская защита",
        "section_title": "Полуоткрытые дебюты",
        "theory": (
            "Сицилианская защита начинается с 1.e4 c5. Чёрные не симметрично повторяют ход белых, "
            "а сразу борются за центр с фланга. После 2.Nf3 d6 3.d4 позиция становится острой и "
            "асимметричной - у чёрных есть контригра на ферзевом фланге. Это самый частый ответ "
            "чёрных на 1.e4 на высоком уровне."
        ),
        "sort_order": 3,
        "tasks": [
            {
                "task_type": "make_move",
                "title": "Начало сицилианской",
                "instruction": "Сделайте первый ход белыми - e2-e4.",
                "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                "source_square": "e2",
                "difficulty": 1,
                "payload": {"piece": "P", "mode": "accepted_moves", "accepted_moves": ["e2e4"]},
                "explanation": "1.e4 занимает центр и открывает игру.",
                "sort_order": 1,
            },
            {
                "task_type": "make_move",
                "title": "Мобилизация против c5",
                "instruction": "Чёрные ответили 1...c5. Развивайте коня g1 на f3.",
                "fen": "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
                "source_square": "g1",
                "difficulty": 1,
                "payload": {"piece": "N", "mode": "accepted_moves", "accepted_moves": ["g1f3"]},
                "explanation": "2.Nf3 развивает коня и подготавливает продвижение d2-d4.",
                "sort_order": 2,
            },
            {
                "task_type": "make_move",
                "title": "Прорыв центра",
                "instruction": "После 2...d6 сыграйте d2-d4 - главный ход против сицилианской.",
                "fen": "rnbqkbnr/pp2pppp/3p4/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 3",
                "source_square": "d2",
                "difficulty": 2,
                "payload": {"piece": "P", "mode": "accepted_moves", "accepted_moves": ["d2d4"]},
                "explanation": "3.d4 раскрывает центр; после размена пешек белые получают пространственное преимущество.",
                "sort_order": 3,
            },
        ],
    },
    {
        "slug": "french",
        "title": "Французская защита",
        "section_title": "Полуоткрытые дебюты",
        "theory": (
            "Французская защита образуется ходами 1.e4 e6 2.d4 d5. Чёрные держат свой ход d7-d5 "
            "в качестве ответа ферзевой пешке белых и создают прочную пешечную цепь. После 3.e5 "
            "у белых пространство на королевском фланге, а у чёрных - надёжное пешечное остов и "
            "контригра на ферзевом фланге."
        ),
        "sort_order": 4,
        "tasks": [
            {
                "task_type": "make_move",
                "title": "Первый ход французской",
                "instruction": "Сделайте первый ход белыми - e2-e4.",
                "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                "source_square": "e2",
                "difficulty": 1,
                "payload": {"piece": "P", "mode": "accepted_moves", "accepted_moves": ["e2e4"]},
                "explanation": "1.e4 открывает партию.",
                "sort_order": 1,
            },
            {
                "task_type": "make_move",
                "title": "Ферзевая пешка в центр",
                "instruction": "После 1...e6 сыграйте d2-d4.",
                "fen": "rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
                "source_square": "d2",
                "difficulty": 1,
                "payload": {"piece": "P", "mode": "accepted_moves", "accepted_moves": ["d2d4"]},
                "explanation": "2.d4 занимает сильное пешечное поле в центре.",
                "sort_order": 2,
            },
            {
                "task_type": "make_move",
                "title": "Пешечная цепь",
                "instruction": "Чёрные ответили 2...d5. Продвиньте пешку e4 на e5.",
                "fen": "rnbqkbnr/ppp2ppp/4p3/3p4/3PP3/8/PPP2PPP/RNBQKBNR w KQkq - 0 3",
                "source_square": "e4",
                "difficulty": 2,
                "payload": {"piece": "P", "mode": "accepted_moves", "accepted_moves": ["e4e5"]},
                "explanation": "3.e5 строит пешечную цепь с поддержкой в центре и отступает пешке d5 назад нельзя.",
                "sort_order": 3,
            },
        ],
    },
    {
        "slug": "queens-gambit",
        "title": "Ферзевый гамбит",
        "section_title": "Закрытые дебюты",
        "theory": (
            "Ферзевый гамбит начинается ходами 1.d4 d5 2.c4. Белые предлагают пешку в обмен на "
            "контроль над центром при ходе 2...dxc4. Если чёрные принимают гамбит, белые развиваются "
            "с большим комфортом и получают давление по центральным линиям. Обычно чёрные либо "
            "принимают, либо удерживают центр ходом 2...e6."
        ),
        "sort_order": 5,
        "tasks": [
            {
                "task_type": "make_move",
                "title": "Начало ферзевого гамбита",
                "instruction": "Сделайте первый ход белыми - d2-d4.",
                "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                "source_square": "d2",
                "difficulty": 1,
                "payload": {"piece": "P", "mode": "accepted_moves", "accepted_moves": ["d2d4"]},
                "explanation": "1.d4 борется за центральные поля d5 и e5.",
                "sort_order": 1,
            },
            {
                "task_type": "make_move",
                "title": "Предложение пешки",
                "instruction": "После 1...d5 сыграйте c2-c4.",
                "fen": "rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 2",
                "source_square": "c2",
                "difficulty": 2,
                "payload": {"piece": "P", "mode": "accepted_moves", "accepted_moves": ["c2c4"]},
                "explanation": "2.c4 атакует пешку d5 и предлагает чёрным принять гамбит ходом 2...dxc4.",
                "sort_order": 2,
            },
            {
                "task_type": "make_move",
                "title": "Развитие коня",
                "instruction": "Чёрные укрепили центр ходом 2...e6. Развивайте коня b1 на c3.",
                "fen": "rnbqkbnr/ppp2ppp/4p3/3p4/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3",
                "source_square": "b1",
                "difficulty": 2,
                "payload": {"piece": "N", "mode": "accepted_moves", "accepted_moves": ["b1c3"]},
                "explanation": "3.Nc3 развивает коня и поддерживает пешку c4.",
                "sort_order": 3,
            },
        ],
    },
    {
        "slug": "english",
        "title": "Английское начало",
        "section_title": "Закрытые дебюты",
        "theory": (
            "Английское начало открывается ходом 1.c4. Белые контролируют поле d5 и готовятся "
            "к игре на ферзевом фланге ещё до занятия центра пешками. После 1...e5 2.Nc3 белые "
            "развивают коня и создают давление на пешку e5 - классическая борьба за центральные поля."
        ),
        "sort_order": 6,
        "tasks": [
            {
                "task_type": "make_move",
                "title": "Первый ход англичан",
                "instruction": "Сделайте первый ход белыми - c2-c4.",
                "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                "source_square": "c2",
                "difficulty": 1,
                "payload": {"piece": "P", "mode": "accepted_moves", "accepted_moves": ["c2c4"]},
                "explanation": "1.c4 занимает ферзевый фланг и контролирует поле d5.",
                "sort_order": 1,
            },
            {
                "task_type": "make_move",
                "title": "Нажим на пешку e5",
                "instruction": "Чёрные ответили 1...e5. Развивайте коня b1 на c3.",
                "fen": "rnbqkbnr/pppp1ppp/8/4p3/2P5/8/PP1PPPPP/RNBQKBNR w KQkq - 0 2",
                "source_square": "b1",
                "difficulty": 2,
                "payload": {"piece": "N", "mode": "accepted_moves", "accepted_moves": ["b1c3"]},
                "explanation": "2.Nc3 развивает коня и давит на пешку e5, не давая чёрным развиваться по стандартной схеме.",
                "sort_order": 2,
            },
        ],
    },
    {
        "slug": "scotch",
        "title": "Шотландская партия",
        "section_title": "Открытые дебюты",
        "theory": (
            "Шотландская партия начинается ходами 1.e4 e5 2.Nf3 Nc6 3.d4. Белые сразу вскрывают "
            "центр, а не развиваются спокойно, как в итальянской или испанской партии. После "
            "3...exd4 4.Nxd4 конь занимает центральное поле d4, откуда контролирует важные поля. "
            "Шотландская партия даёт белым быструю и удобную игру."
        ),
        "sort_order": 7,
        "tasks": [
            {
                "task_type": "make_move",
                "title": "Начните шотландскую",
                "instruction": "Чёрные ответили 2...Nc6. Вскройте центр ходом 3.d4.",
                "fen": "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
                "source_square": "d2",
                "difficulty": 2,
                "payload": {"piece": "P", "mode": "accepted_moves", "accepted_moves": ["d2d4"]},
                "explanation": "3.d4 атакует пешку e5 и открывает линии для белых фигур.",
                "sort_order": 1,
            },
            {
                "task_type": "make_move",
                "title": "Возьмите на d4",
                "instruction": "Чёрные сыграли 3...exd4. Верните пешку ходом коня f3.",
                "fen": "r1bqkbnr/pppp1ppp/2n5/8/3pP3/5N2/PPP2PPP/RNBQKB1R w KQkq - 0 4",
                "source_square": "f3",
                "difficulty": 2,
                "payload": {"piece": "N", "mode": "accepted_moves", "accepted_moves": ["f3d4"]},
                "explanation": "4.Nxd4 ставит коня в центр на d4 и возвращает пешку.",
                "sort_order": 2,
            },
            {
                "task_type": "choose_option",
                "title": "Идея шотландской",
                "instruction": "В чём главная идея хода 3.d4 в шотландской партии?",
                "fen": "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
                "source_square": "d2",
                "difficulty": 2,
                "payload": {
                    "options": _options(
                        ("center", "Сразу вскрыть центр и получить фигурам простор"),
                        ("castle", "Подготовить рокировку"),
                        ("trap", "Поймать слона в ловушку"),
                        ("trade_queens", "Разменять ферзей"),
                    ),
                    "correct_option": "center",
                },
                "explanation": "Белые стремятся к открытой игре с быстрым развитием и давлением в центре.",
                "sort_order": 3,
            },
        ],
    },
    {
        "slug": "caro-kann",
        "title": "Защита Каро-Канн",
        "section_title": "Полуоткрытые дебюты",
        "theory": (
            "Защита Каро-Канн начинается ходами 1.e4 c6. Чёрные готовят 2...d5 и поддерживают пешку "
            "d5 ходом c6. В отличие от французской защиты, слон c8 остаётся на свободе. Чёрные "
            "получают прочную, но несколько пассивную позицию и часто играют на равную борьбу."
        ),
        "sort_order": 7,
        "tasks": [
            {
                "task_type": "make_move",
                "title": "Первый ход белых",
                "instruction": "Сделайте ход 1.e4.",
                "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                "source_square": "e2",
                "difficulty": 1,
                "payload": {"piece": "P", "mode": "accepted_moves", "accepted_moves": ["e2e4"]},
                "explanation": "1.e4 занимает центр и открывает выход слону f1 и ферзю d1.",
                "sort_order": 1,
            },
            {
                "task_type": "make_move",
                "title": "Занимаем центр",
                "instruction": "Чёрные ответили 1...c6, готовя d5. Заняните центр ходом 2.d4.",
                "fen": "rnbqkbnr/pp1ppppp/2p5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
                "source_square": "d2",
                "difficulty": 2,
                "payload": {"piece": "P", "mode": "accepted_moves", "accepted_moves": ["d2d4"]},
                "explanation": "2.d4 создаёт сильный пешечный центр и не даёт чёрным легко играть d5.",
                "sort_order": 2,
            },
            {
                "task_type": "make_move",
                "title": "Развитие коня",
                "instruction": "Чёрные сыграли 2...d5. Развейте коня b1 на c3.",
                "fen": "rnbqkbnr/pp2pppp/2p5/3p4/3PP3/8/PPP2PPP/RNBQKBNR w KQkq - 0 3",
                "source_square": "b1",
                "difficulty": 2,
                "payload": {"piece": "N", "mode": "accepted_moves", "accepted_moves": ["b1c3"]},
                "explanation": "3.Nc3 защищает пешку e4 и развивает коня к центру.",
                "sort_order": 3,
            },
            {
                "task_type": "choose_option",
                "title": "Идея Каро-Канна",
                "instruction": "Зачем чёрные в защите Каро-Канн играют c7-c6?",
                "fen": "rnbqkbnr/pp1ppppp/2p5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
                "source_square": "d2",
                "difficulty": 2,
                "payload": {
                    "options": _options(
                        ("support_d5", "Чтобы поддержать пешкой ход d7-d5"),
                        ("attack_e4", "Чтобы сразу атаковать пешку e4"),
                        ("fianchetto", "Чтобы вывести слона на g7"),
                        ("castle_early", "Чтобы быстрее рокироваться"),
                    ),
                    "correct_option": "support_d5",
                },
                "explanation": "Пешка c6 поддерживает продвижение d7-d5, оставляя слону c8 открытую дорогу.",
                "sort_order": 4,
            },
        ],
    },
    {
        "slug": "nimzo-indian",
        "title": "Защита Нимцовича",
        "section_title": "Закрытые дебюты",
        "theory": (
            "Защита Нимцовича возникает после 1.d4 Nf6 2.c4 e6 3.Nc3 Bb4. Чёрные развивают слона на b4 "
            "и связывают коня c3, борясь за контроль над полем e4. Это один из самых надёжных ответов "
            "на 1.d4: чёрные получают гибкую позицию и часто отдают слона за коня на c3."
        ),
        "sort_order": 7,
        "tasks": [
            {
                "task_type": "make_move",
                "title": "Начните с d4",
                "instruction": "Сделайте ход 1.d4.",
                "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                "source_square": "d2",
                "difficulty": 1,
                "payload": {"piece": "P", "mode": "accepted_moves", "accepted_moves": ["d2d4"]},
                "explanation": "1.d4 занимает центр и открывает диагональ слону c1.",
                "sort_order": 1,
            },
            {
                "task_type": "make_move",
                "title": "Подкрепите центр",
                "instruction": "Чёрные ответили 1...Nf6. Сыграйте 2.c4.",
                "fen": "rnbqkb1r/pppppppp/5n2/8/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 1 2",
                "source_square": "c2",
                "difficulty": 2,
                "payload": {"piece": "P", "mode": "accepted_moves", "accepted_moves": ["c2c4"]},
                "explanation": "2.c4 поддерживает пешку d4 и захватывает ещё больше пространства в центре.",
                "sort_order": 2,
            },
            {
                "task_type": "make_move",
                "title": "Развитие коня",
                "instruction": "Чёрные сыграли 2...e6. Развейте коня b1 на c3.",
                "fen": "rnbqkb1r/pppp1ppp/4pn2/8/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3",
                "source_square": "b1",
                "difficulty": 2,
                "payload": {"piece": "N", "mode": "accepted_moves", "accepted_moves": ["b1c3"]},
                "explanation": "3.Nc3 защищает пешку d4 и готовит развитие остальных фигур.",
                "sort_order": 3,
            },
            {
                "task_type": "choose_option",
                "title": "Идея защиты Нимцовича",
                "instruction": "Почему в защите Нимцовича чёрные часто играют Bxc3?",
                "fen": "rnbqkb1r/pppp1ppp/4pn2/8/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3",
                "source_square": "b1",
                "difficulty": 3,
                "payload": {
                    "options": _options(
                        ("pawns", "Чтобы испортить белым пешечную структуру"),
                        ("mate", "Чтобы сразу поставить мат"),
                        ("rook", "Чтобы выиграть ладью"),
                        ("castle", "Чтобы получить право рокировки"),
                    ),
                    "correct_option": "pawns",
                },
                "explanation": "Размен Bxc3 портит белым пешки, и чёрные получают долговременные шансы против центра.",
                "sort_order": 4,
            },
        ],
    },
]


ENDGAMES_LESSONS = [
    {
        "slug": "king-rook-mate",
        "title": "Мат ладьёй и королём",
        "section_title": "Мат тяжёлыми фигурами",
        "theory": (
            "Мат ладьёй и королём - базовое окончание. Одинокого короля соперника оттесняют к краю доски: "
            "ладья отрезает горизонтали, а свой король подходит к сопернику и закрывает поля отступления. "
            "Мат объявляется ладьёй на последней линии при поддержке своего короля."
        ),
        "sort_order": 1,
        "tasks": [
            {
                "task_type": "make_move",
                "title": "Мат ладьёй в один ход",
                "instruction": "Сделайте ход ладьёй f7, который объявляет мат чёрному королю.",
                "fen": "n1B4k/5R2/2p3K1/P1N5/1pP4p/4r3/P5Pb/3q4 w - - 0 25",
                "source_square": "f7",
                "difficulty": 2,
                "payload": {"piece": "R", "mode": "accepted_moves", "accepted_moves": ["f7f8"]},
                "explanation": "Ладья встаёт на f8 и матует короля h8: восьмая горизонталь отрезана ладьёй, а поля g7 и h7 перекрывает король g6.",
                "sort_order": 1,
            },
            {
                "task_type": "choose_option",
                "title": "Почему мат ставится на краю?",
                "instruction": "Почему мат ладьёй и королём ставится на краю доски?",
                "fen": "n1B4k/5R2/2p3K1/P1N5/1pP4p/4r3/P5Pb/3q4 w - - 0 25",
                "source_square": None,
                "difficulty": 1,
                "payload": {
                    "options": _options(
                        ("edge", "У одинокого короля возле края меньше полей для отступления"),
                        ("center", "Ладья сильнее всего в центре доски"),
                        ("always", "Мат ладьёй возможен в любой точке доски"),
                    ),
                    "correct_option": "edge",
                },
                "explanation": "В центре одинокий король может бежать в любую сторону; у самого края его отступление практически полностью перекрыто ладьёй и королём.",
                "sort_order": 2,
            },
            {
                "task_type": "make_move",
                "title": "Мат ладьёй на последней линии",
                "instruction": "Сделайте ход ладьёй h7, объявляющий мат чёрному королю a8.",
                "fen": "k7/6BR/1K1p2N1/3p4/pp1P1P2/3RP1n1/1P4b1/1Q6 w - - 0 23",
                "source_square": "h7",
                "difficulty": 2,
                "payload": {"piece": "R", "mode": "accepted_moves", "accepted_moves": ["h7h8"]},
                "explanation": "Ладья встаёт на h8 и отрезает восьмую горизонталь: король a8 не может уйти ни на a7 или b7 — их держит король b6, — ни на b8, который контролирует ладья.",
                "sort_order": 3,
            },
        ],
    },
    {
        "slug": "king-queen-mate",
        "title": "Мат ферзём и королём",
        "section_title": "Мат тяжёлыми фигурами",
        "theory": (
            "Ферзь вместе с королём матует одинокого короля быстро и надёжно: ферзь отсекает поля, "
            "а король поддерживает его. Главная ошибка в таком окончании - поставить пат: если зажатый "
            "король не под шахом, но ходить ему некуда, вместо победы наступает ничья."
        ),
        "sort_order": 2,
        "tasks": [
            {
                "task_type": "make_move",
                "title": "Мат ферзём в один ход",
                "instruction": "Сделайте ход ферзём e3, который объявляет мат чёрному королю.",
                "fen": "k6b/1pK2p1p/4p2B/6P1/4PP1n/4Q2P/3R3N/1r6 w - - 0 40",
                "source_square": "e3",
                "difficulty": 2,
                "payload": {"piece": "Q", "mode": "accepted_moves", "accepted_moves": ["e3a3"]},
                "explanation": "Ферзь становится на a3 и матует короля a8 по вертикали a: поля a7, b7 и b8 закрыты ферзём и королём c7.",
                "sort_order": 1,
            },
            {
                "task_type": "choose_option",
                "title": "Чего нужно опасаться?",
                "instruction": "Какую ошибку чаще всего допускают при мате ферзём и королём?",
                "fen": "7k/2r5/K2p4/6q1/4Q3/N6b/Pn3P1P/8 b - - 0 36",
                "source_square": None,
                "difficulty": 2,
                "payload": {
                    "options": _options(
                        ("pat", "Поставить пат и заменить победу ничьей"),
                        ("loss", "Отдать ферзя за ладью и остаться без мата"),
                        ("time", "Затянуть партию и дожидаться сдачи соперника"),
                    ),
                    "correct_option": "pat",
                },
                "explanation": "Когда одинокий король зажат, легко ограничить его без шаха: получится пат - ничья вместо победы.",
                "sort_order": 2,
            },
            {
                "task_type": "make_move",
                "title": "Мат ферзём по последней линии",
                "instruction": "Сделайте ход ферзём h1, объявляющий мат чёрному королю g8.",
                "fen": "6k1/p2p4/3p2KR/NP3P1P/6p1/3r4/1B2nb1P/7Q w - - 0 35",
                "source_square": "h1",
                "difficulty": 2,
                "payload": {"piece": "Q", "mode": "accepted_moves", "accepted_moves": ["h1a8"]},
                "explanation": "Ферзь переходит на a8 и контролирует всю восьмую горизонталь, а король g6 закрывает чёрному королю поля g7, f7 и h7 — мат.",
                "sort_order": 3,
            },
        ],
    },
    {
        "slug": "two-rooks-mate",
        "title": "Мат двумя ладьями",
        "section_title": "Мат тяжёлыми фигурами",
        "theory": (
            "Две ладьи матуют одинокого короля «лестницей»: одна ладья отрезает ряд, "
            "вторая оттесняет короля к краю доски. Ладьи действуют по соседним линиям попеременно, "
            "поэтому король не может ни уйти от шахов, ни приблизиться к ним; мат ставится на краю доски."
        ),
        "sort_order": 3,
        "tasks": [
            {
                "task_type": "make_move",
                "title": "Мат двумя ладьями в один ход",
                "instruction": "Найдите мат в один ход ладьёй с поля b1.",
                "fen": "7k/6R1/p1Np1K2/8/p3pP2/2P1n3/P3P3/1R6 w - - 0 25",
                "source_square": "b1",
                "difficulty": 2,
                "payload": {"piece": "R", "mode": "accepted_moves", "accepted_moves": ["b1b8", "b1h1"]},
                "explanation": "На восьмую горизонталь или на вертикаль h — ладьи вместе с королём f6 перекрывают все поля отступления, и чёрный король h8 получает мат.",
                "sort_order": 1,
            },
            {
                "task_type": "choose_option",
                "title": "Как работает «лестница»?",
                "instruction": "Почему две ладьи легко матуют одинокого короля?",
                "fen": "7k/6R1/p1Np1K2/8/p3pP2/2P1n3/P3P3/1R6 w - - 0 25",
                "source_square": None,
                "difficulty": 2,
                "payload": {
                    "options": _options(
                        ("ladder", "Ладьи по очереди отрезают ряды и теснят короля к краю"),
                        ("side", "Ладьи должны стоять рядом, клетка к клетке"),
                        ("center", "Ладьи сильны только в центре доски"),
                    ),
                    "correct_option": "ladder",
                },
                "explanation": "Одна ладья отрезает ряд, вторая даёт шах и оттесняет короля; так его шаг за шагом гонят к краю доски, где и ставят мат.",
                "sort_order": 2,
            },
        ],
    },
    {
        "slug": "king-and-pawn",
        "title": "Король ведёт пешку",
        "section_title": "Пешки и оппозиция",
        "theory": (
            "Проходная пешка проходит в ферзи при поддержке собственного короля. Король должен идти "
            "впереди пешки: он занимает ключевые поля, оттесняет короля соперника и защищает пешку "
            "на всём пути к последней горизонтали."
        ),
        "sort_order": 4,
        "tasks": [
            {
                "task_type": "make_move",
                "title": "Проведите пешку с темпом",
                "instruction": "Сделайте ход пешкой d5, продвигая её под защитой короля e5.",
                "fen": "8/1p2k1N1/7R/3PK1p1/1Pp3p1/r7/2P1PbP1/5n2 w - - 0 36",
                "source_square": "d5",
                "difficulty": 2,
                "payload": {"piece": "P", "mode": "accepted_moves", "accepted_moves": ["d5d6"]},
                "explanation": "Пешка входит на d6 под защитой короля e5 и с шахом отбрасывает чёрного короля; пешке остаётся два поля до превращения.",
                "sort_order": 1,
            },
            {
                "task_type": "choose_option",
                "title": "Роль короля",
                "instruction": "Зачем король сопровождает проходную пешку?",
                "fen": "8/1p2k1N1/7R/3PK1p1/1Pp3p1/r7/2P1PbP1/5n2 w - - 0 36",
                "source_square": None,
                "difficulty": 1,
                "payload": {
                    "options": _options(
                        ("lead", "Занимать поля перед пешкой и оттеснять короля соперника"),
                        ("back", "Оставаться позади и не вмешиваться в игру"),
                        ("trade", "Поскорее разменять пешку на лёгкую фигуру"),
                    ),
                    "correct_option": "lead",
                },
                "explanation": "Король идёт вместе с пешкой: он защищает её и не пускает короля соперника к полям превращения.",
                "sort_order": 2,
            },
            {
                "task_type": "choose_option",
                "title": "Где должен идти король?",
                "instruction": "Как король сопровождает проходную пешку к превращению?",
                "fen": "8/1p2k1N1/7R/3PK1p1/1Pp3p1/r7/2P1PbP1/5n2 w - - 0 36",
                "source_square": None,
                "difficulty": 1,
                "payload": {
                    "options": _options(
                        ("front", "Впереди пешки, занимая ключевые поля"),
                        ("behind", "Далеко позади пешки, не мешая ей"),
                        ("corner", "В углу доски, вдали от пешки"),
                    ),
                    "correct_option": "front",
                },
                "explanation": "Король идёт впереди пешки: он оттесняет короля соперника и защищает пешку, освобождая ей путь к полю превращения.",
                "sort_order": 3,
            },
        ],
    },
    {
        "slug": "pawn-square",
        "title": "Правило квадрата",
        "section_title": "Пешки и оппозиция",
        "theory": (
            "Правило квадрата быстро решает, успеет ли король соперника догнать проходную пешку. "
            "Проведите от пешки диагональ до последней линии и постройте воображаемый квадрат: если король "
            "соперника уже внутри квадрата - он догонит пешку; если король за его пределами и очередь хода "
            "за пешкой - пешка проходит в ферзи сама."
        ),
        "sort_order": 5,
        "tasks": [
            {
                "task_type": "choose_option",
                "title": "Король вне квадрата",
                "instruction": "Белая пешка c4, чёрный король g3, ход белых. Догонит ли чёрный король пешку?",
                "fen": "3bn3/5R2/p1p4p/8/1pPP2k1/8/2BPP2P/K6N w - - 0 28",
                "source_square": None,
                "difficulty": 2,
                "payload": {
                    "options": _options(
                        ("no", "Нет: король вне квадрата, пешка проходит в ферзи"),
                        ("yes", "Да: король успеет перекрыть путь"),
                        ("draw", "Ничья при любом продолжении"),
                    ),
                    "correct_option": "no",
                },
                "explanation": "Квадрат пешки c4 - поля c4–c8–g8–g4, а король g3 стоит за его пределами; с ходом белых пешка убегает в ферзи.",
                "sort_order": 1,
            },
            {
                "task_type": "make_move",
                "title": "Пешка убегает вперёд",
                "instruction": "Продвиньте пешку c4 так, чтобы чёрный король остался за пределами квадрата.",
                "fen": "3bn3/5R2/p1p4p/8/1pPP2k1/8/2BPP2P/K6N w - - 0 28",
                "source_square": "c4",
                "difficulty": 2,
                "payload": {"piece": "P", "mode": "accepted_moves", "accepted_moves": ["c4c5"]},
                "explanation": "Ход c5 держит чёрного короля вне нового квадрата c5–c8–g8–g5, и пешка беспрепятственно шагает к превращению.",
                "sort_order": 2,
            },
        ],
    },
    {
        "slug": "pawn-breakthrough",
        "title": "Пешечный прорыв",
        "section_title": "Пешки и оппозиция",
        "theory": (
            "Пешечный прорыв — это жертва пешек ради того, чтобы одна из них прошла в ферзи. "
            "Когда своих пешек на участке больше, чем у соперника, цепь подрывают ходом-жертвой: "
            "соперник обязан взять, а следующая пешка идёт вперёд и вскрывает оборону."
        ),
        "sort_order": 6,
        "tasks": [
            {
                "task_type": "make_move",
                "title": "Начните пешечный прорыв",
                "instruction": "Сделайте ход пешкой b5, начиная прорыв пешечной цепи.",
                "fen": "6k1/ppp5/5r2/PPP5/ppp3bn/4R3/PPP5/1N5K w - - 0 25",
                "source_square": "b5",
                "difficulty": 3,
                "payload": {"piece": "P", "mode": "accepted_moves", "accepted_moves": ["b5b6"]},
                "explanation": "Ход b6! подрывает пешечную цепь чёрных: после взятия на b6 следует c6! а затем a6! — и одна из белых пешек беспрепятственно проходит в ферзи.",
                "sort_order": 1,
            },
            {
                "task_type": "choose_option",
                "title": "В чём идея прорыва?",
                "instruction": "Что такое пешечный прорыв?",
                "fen": "6k1/ppp5/5r2/PPP5/ppp3bn/4R3/PPP5/1N5K w - - 0 25",
                "source_square": None,
                "difficulty": 2,
                "payload": {
                    "options": _options(
                        ("sac", "Отдать пешки, чтобы одна из них прошла в ферзи"),
                        ("trade", "Разменять все пешки и получить ничью"),
                        ("wait", "Не двигать пешки и ждать ошибки соперника"),
                    ),
                    "correct_option": "sac",
                },
                "explanation": "Смысл прорыва — пожертвовать часть пешек, чтобы освободить дорогу другой: она беспрепятственно доходит до последней горизонтали.",
                "sort_order": 2,
            },
        ],
    },
    {
        "slug": "en-passant",
        "title": "Взятие на проходе",
        "section_title": "Пешки и оппозиция",
        "theory": (
            "Если пешка соперника с начальной клетки проходит сразу две клетки и оказывается рядом с вашей пешкой, "
            "её можно взять на проходе так, будто она прошла только одну клетку. Такое право существует только немедленно следующим ходом."
        ),
        "sort_order": 7,
        "tasks": [
            {
                "task_type": "make_move",
                "title": "Возьмите на проходе",
                "instruction": "Выполните белой пешкой e5 взятие на проходе.",
                "fen": "4n2k/pp2p3/4N3/b2pP2P/6p1/P3P1P1/2Q5/K4B2 w - d6 0 20",
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
                "fen": "4n2k/pp2p3/4N3/b2pP2P/6p1/P3P1P1/2Q5/K4B2 w - d6 0 20",
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
        "section_title": "Пешки и оппозиция",
        "theory": (
            "Дойдя до последней горизонтали, пешка обязана превратиться в ферзя, ладью, слона или коня. "
            "Выбор новой фигуры не зависит от того, есть ли такая фигура уже на доске."
        ),
        "sort_order": 8,
        "tasks": [
            {
                "task_type": "select_squares",
                "title": "Поле превращения",
                "instruction": "Отметьте клетку, на которую должна пойти белая пешка e7 для превращения.",
                "fen": "1B5k/4P3/2p1ppb1/P1p5/2P5/5P1P/6N1/K4n2 w - - 0 26",
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
                "fen": "1B5k/4P3/2p1ppb1/P1p5/2P5/5P1P/6N1/K4n2 w - - 0 26",
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
    {
        "slug": "opposition",
        "title": "Оппозиция королей",
        "section_title": "Пешки и оппозиция",
        "theory": (
            "Оппозиция - положение королей друг напротив друга через одну клетку. Владеет оппозицией тот, "
            "кто не ходит: соперник вынужден отступать королём и уступать дорогу, поэтому ваша пешка "
            "получает решающую поддержку на пути к превращению."
        ),
        "sort_order": 9,
        "tasks": [
            {
                "task_type": "choose_option",
                "title": "Что такое оппозиция?",
                "instruction": "Какое расположение королей называют оппозицией?",
                "fen": "6R1/2p5/4k2B/n1P4p/pp2K2b/8/1P2P2P/7N w - - 0 40",
                "source_square": None,
                "difficulty": 1,
                "payload": {
                    "options": _options(
                        ("one", "Короли стоят друг напротив друга через одну клетку"),
                        ("side", "Короли стоят рядом, клетка к клетке"),
                        ("far", "Короли стоят в противоположных углах доски"),
                    ),
                    "correct_option": "one",
                },
                "explanation": "Оппозиция: два короля на одной линии через одну клетку - они не могут стоять к клетке, поэтому между ними всегда ровно одно поле.",
                "sort_order": 1,
            },
            {
                "task_type": "make_move",
                "title": "Продвижение с шахом",
                "instruction": "Сделайте ход пешкой d4, который поддержан королём e4 и даёт шах чёрному королю.",
                "fen": "q3b3/np5p/p1p1k3/B5Q1/3PKP1N/1P1P1P2/1R6/2r5 w - - 0 33",
                "source_square": "d4",
                "difficulty": 3,
                "payload": {"piece": "P", "mode": "accepted_moves", "accepted_moves": ["d4d5"]},
                "explanation": "Пешка идёт на d5 с шахом: чёрный король обязан отступить, а пешка продолжает путь, защищённая королём e4.",
                "sort_order": 2,
            },
        ],
    },
    {
        "slug": "queen-mate-avoid-stalemate",
        "title": "Ферзь и король: осторожно с патом",
        "section_title": "Мат тяжёлыми фигурами",
        "theory": (
            "Ферзь и король легко ставят мат одинокому королю, но здесь легко по ошибке сделать пат: "
            "ферзь отнимает у короля сразу много полей. Приём простой - сначала оттеснить короля "
            "краем доски, оставляя ему хотя бы одно поле, и лишь затем объявить мат. Пат при большом "
            "перевесе - обидная ничья."
        ),
        "sort_order": 10,
        "tasks": [
            {
                "task_type": "make_move",
                "title": "Поставьте мат ферзём",
                "instruction": "Чёрный король на h8. Поставьте мат ходом ферзя на d8, не допустив пата.",
                "fen": "1b5k/4R3/p5K1/2P5/7p/NP3B1P/3Q4/8 w - - 0 36",
                "source_square": "d2",
                "difficulty": 3,
                "payload": {"piece": "Q", "mode": "accepted_moves", "accepted_moves": ["d2d8"]},
                "explanation": "Qd8 бьёт короля по последней горизонтали, а собственный король g6 держит поля g7 и h7 - это мат.",
                "sort_order": 1,
            },
            {
                "task_type": "choose_option",
                "title": "Не сделайте пат",
                "instruction": "О чём нужно помнить, ставя мат одним ферзём?",
                "fen": "1b5k/4R3/p5K1/2P5/7p/NP3B1P/3Q4/8 w - - 0 36",
                "source_square": "d2",
                "difficulty": 3,
                "payload": {
                    "options": _options(
                        ("stalemate", "Остерегаться пата: не отнимать у короля последнее поле, пока нет мата"),
                        ("check_always", "Каждым ходом объявлять шах"),
                        ("pawn", "Продвигать пешку"),
                        ("repeat", "Повторять позицию до ничьей"),
                    ),
                    "correct_option": "stalemate",
                },
                "explanation": "Ферзь отнимает много полей сразу, поэтому легко ошибиться и поставить не мат, а пат.",
                "sort_order": 2,
            },
        ],
    },
]

TACTICS_LESSONS = [
    {
        "slug": "hanging-piece",
        "title": "Незащищённая фигура",
        "section_title": "Тактические слабости",
        "theory": (
            "Незащищённая фигура — это фигура, которую не прикрывает ни одна своя фигура. "
            "Если её можно забрать без потерь, перед нами готовая тактическая добыча. "
            "Перед нападением проверяйте, какие фигуры соперника остались без защиты."
        ),
        "sort_order": 1,
        "tasks": [
            {
                "task_type": "make_move",
                "title": "Заберите незащищённого ферзя",
                "instruction": "Ферзь чёрных на d5 никем не защищён. Заберите его ферзём с e4.",
                "fen": "6k1/N1p5/p4p2/3q4/4Q3/7R/1r6/n3K3 w - - 0 22",
                "source_square": "e4",
                "difficulty": 1,
                "payload": {
                    "piece": "Q",
                    "mode": "accepted_moves",
                    "accepted_moves": ["e4d5"],
                },
                "explanation": "Ферзь d5 ничем не защищён, поэтому Qxd5 выигрывает материал.",
                "sort_order": 1,
            },
            {
                "task_type": "choose_option",
                "title": "Что такое незащищённая фигура",
                "instruction": "Выберите определение незащищённой фигуры.",
                "fen": "6k1/N1p5/p4p2/3q4/4Q3/7R/1r6/n3K3 w - - 0 22",
                "difficulty": 1,
                "payload": {
                    "options": _options(
                        ("hanging", "Фигура, которую не защищает ни одна своя фигура"),
                        ("pinned", "Фигура, прикрывающая короля"),
                        ("blocked", "Фигура, которой некуда ходить"),
                        ("promoted", "Фигура, прошедшая в ферзи"),
                    ),
                    "correct_option": "hanging",
                },
                "explanation": "Именно отсутствие защиты делает фигуру уязвимой для взятия.",
                "sort_order": 2,
            },
        ],
    },
    {
        "slug": "double-attack",
        "title": "Двойной удар",
        "section_title": "Двойной удар и вилка",
        "theory": (
            "Двойной удар — это ход, после которого одна фигура одновременно нападает "
            "на две цели. Соперник успевает спасти только одну из них, поэтому двойной "
            "удар часто выигрывает материал."
        ),
        "sort_order": 2,
        "tasks": [
            {
                "task_type": "make_move",
                "title": "Конь нападает на короля и ладью",
                "instruction": "Поставьте коня на c7: он объявит шах королю e8 и одновременно нападёт на ладью a8.",
                "fen": "r3k3/7p/8/3N1Npp/2r5/5P1P/1q6/4K1bn w - - 0 35",
                "source_square": "d5",
                "difficulty": 2,
                "payload": {
                    "piece": "N",
                    "mode": "accepted_moves",
                    "accepted_moves": ["d5c7"],
                },
                "explanation": "Nc7+ — двойной удар: король обязан ответить на шах, а ладья a8 остаётся без защиты.",
                "sort_order": 1,
            },
            {
                "task_type": "choose_option",
                "title": "Признак двойного удара",
                "instruction": "Выберите признак двойного удара.",
                "fen": "r3k3/7p/8/3N1Npp/2r5/5P1P/1q6/4K1bn w - - 0 35",
                "difficulty": 1,
                "payload": {
                    "options": _options(
                        ("two_targets", "Одна фигура одновременно нападает на две цели"),
                        ("two_moves", "Две фигуры нападают на одну цель"),
                        ("sacrifice", "Фигура жертвуется ради атаки"),
                        ("defense", "Фигура защищает две свои фигуры"),
                    ),
                    "correct_option": "two_targets",
                },
                "explanation": "Двойной удар ценен тем, что соперник не может закрыть обе угрозы сразу.",
                "sort_order": 2,
            },
        ],
    },
    {
        "slug": "fork",
        "title": "Вилка",
        "section_title": "Двойной удар и вилка",
        "theory": (
            "Вилка — частный случай двойного удара, чаще всего конём или пешкой. "
            "Фигура нападает сразу на две цели, и одна из них обычно теряется."
        ),
        "sort_order": 3,
        "tasks": [
            {
                "task_type": "make_move",
                "title": "Вилка конём",
                "instruction": "Сходите конём на f7: он объявит шах королю h8 и нападёт на ладью d8.",
                "fen": "3r3k/p3p3/8/Pp4NP/5pn1/P7/2Q5/4K2N w - - 0 38",
                "source_square": "g5",
                "difficulty": 2,
                "payload": {
                    "piece": "N",
                    "mode": "accepted_moves",
                    "accepted_moves": ["g5f7"],
                },
                "explanation": "Nf7+ — вилка: король вынужден уйти, после чего конь забирает ладью d8.",
                "sort_order": 1,
            },
            {
                "task_type": "choose_option",
                "title": "Чем вилка отличается от двойного удара",
                "instruction": "Выберите верное утверждение о вилке.",
                "fen": "3r3k/p3p3/8/Pp4NP/5pn1/P7/2Q5/4K2N w - - 0 38",
                "difficulty": 2,
                "payload": {
                    "options": _options(
                        ("special", "Вилка — частный случай двойного удара"),
                        ("same", "Это полностью разные приёмы"),
                        ("defense", "Вилка всегда защищает фигуры"),
                        ("mate", "Вилка всегда заканчивается матом"),
                    ),
                    "correct_option": "special",
                },
                "explanation": "Вилка — разновидность двойного удара, обычно конём или пешкой.",
                "sort_order": 2,
            },
        ],
    },
    {
        "slug": "pin",
        "title": "Связка",
        "section_title": "Связка",
        "theory": (
            "Связка возникает, когда фигура не может уйти, потому что за ней стоит "
            "более ценная фигура или король. Связанную фигуру удобно атаковать: "
            "она либо неподвижна, либо вынуждена пропустить удар."
        ),
        "sort_order": 4,
        "tasks": [
            {
                "task_type": "make_move",
                "title": "Свяжите коня",
                "instruction": "Поставьте слона на b5, чтобы связать коня c6 с королём e8.",
                "fen": "4k2N/2p5/2n2p2/2B4P/B1b1p1p1/2P3P1/6P1/2n1K3 w - - 0 35",
                "source_square": "a4",
                "difficulty": 2,
                "payload": {
                    "piece": "B",
                    "mode": "accepted_moves",
                    "accepted_moves": ["a4b5"],
                },
                "explanation": "Bb5 связывает коня c6: уйти с диагонали b5-e8 он не может, не подставив короля.",
                "sort_order": 1,
            },
            {
                "task_type": "choose_option",
                "title": "Почему фигура связана",
                "instruction": "Выберите, почему конь c6 считается связанным.",
                "fen": "4k2N/2p5/2n2p2/2B4P/B1b1p1p1/2P3P1/6P1/2n1K3 w - - 0 35",
                "difficulty": 1,
                "payload": {
                    "options": _options(
                        ("behind", "За ней находится более ценная фигура или король"),
                        ("blocked", "Ей мешают свои же фигуры"),
                        ("edge", "Она стоит на краю доски"),
                        ("pinned_self", "Она сама нападает на короля"),
                    ),
                    "correct_option": "behind",
                },
                "explanation": "Связанная фигура не может уйти, потому что откроет удар по фигуре за ней.",
                "sort_order": 2,
            },
        ],
    },
    {
        "slug": "capturing-defender",
        "title": "Уничтожение защитника",
        "section_title": "Уничтожение защитника",
        "theory": (
            "Иногда нужную фигуру нельзя забрать сразу, потому что её защищает другая. "
            "Тогда сначала уничтожают защитника — и цель остаётся без прикрытия."
        ),
        "sort_order": 5,
        "tasks": [
            {
                "task_type": "make_move",
                "title": "Уберите защитника слоном",
                "instruction": "Заберите слоном коня f6 — он прикрывает важные поля вокруг короля.",
                "fen": "4k3/r7/1p1pnn1N/8/2P4P/2B1P3/6BP/1b2K3 w - - 0 40",
                "source_square": "c3",
                "difficulty": 2,
                "payload": {
                    "piece": "B",
                    "mode": "accepted_moves",
                    "accepted_moves": ["c3f6"],
                },
                "explanation": "Bxf6 убирает защитника, после чего нападение на его подопечные поля становится решающим.",
                "sort_order": 1,
            },
            {
                "task_type": "choose_option",
                "title": "Зачем уничтожать защитника",
                "instruction": "Выберите, зачем уничтожают защитника.",
                "fen": "4k3/r7/1p1pnn1N/8/2P4P/2B1P3/6BP/1b2K3 w - - 0 40",
                "difficulty": 1,
                "payload": {
                    "options": _options(
                        ("expose", "Чтобы лишить цель защиты и забрать её"),
                        ("trade", "Чтобы разменять все фигуры"),
                        ("tempo", "Чтобы выиграть темп в дебюте"),
                        ("stalemate", "Чтобы поставить пат"),
                    ),
                    "correct_option": "expose",
                },
                "explanation": "Без защитника цель становится уязвимой для взятия.",
                "sort_order": 2,
            },
        ],
    },
    {
        "slug": "discovered-attack",
        "title": "Открытое нападение",
        "section_title": "Открытое нападение и вскрытый шах",
        "theory": (
            "Открытое нападение — это когда фигура уходит с линии, освобождая дорогу "
            "другой своей фигуре. Уходящая фигура тоже может создавать угрозу, "
            "поэтому сопернику приходится реагировать на две атаки сразу."
        ),
        "sort_order": 6,
        "tasks": [
            {
                "task_type": "make_move",
                "title": "Освободите линию ладье",
                "instruction": "Уведите слона с e4 так, чтобы ладья e1 напала на ферзя e7.",
                "fen": "1n4k1/4q3/1p5p/1N5p/4BP2/1P5P/R7/4R1K1 w - - 0 27",
                "source_square": "e4",
                "difficulty": 2,
                "payload": {
                    "piece": "B",
                    "mode": "accepted_moves",
                    "accepted_moves": ["e4d5"],
                },
                "explanation": "Bd5 освобождает вертикаль e, и ладья e1 нападает на ферзя e7.",
                "sort_order": 1,
            },
            {
                "task_type": "choose_option",
                "title": "Что открывает нападение",
                "instruction": "Выберите, что открывает нападение ладьи.",
                "fen": "1n4k1/4q3/1p5p/1N5p/4BP2/1P5P/R7/4R1K1 w - - 0 27",
                "difficulty": 1,
                "payload": {
                    "options": _options(
                        ("clear_line", "Уход фигуры с линии, по которой бьёт другая фигура"),
                        ("capture", "Взятие незащищённой фигуры"),
                        ("castle", "Рокировка"),
                        ("promotion", "Превращение пешки"),
                    ),
                    "correct_option": "clear_line",
                },
                "explanation": "Открытое нападение работает за счёт освобождения линии для другой фигуры.",
                "sort_order": 2,
            },
        ],
    },
    {
        "slug": "discovered-check",
        "title": "Вскрытый шах",
        "section_title": "Открытое нападение и вскрытый шах",
        "theory": (
            "Вскрытый шах — особый случай открытого нападения: уходящая фигура "
            "открывает шах по королю. Соперник обязан отвечать на шах, и у вас "
            "появляется свободный ход для второй угрозы."
        ),
        "sort_order": 7,
        "tasks": [
            {
                "task_type": "make_move",
                "title": "Объявите вскрытый шах",
                "instruction": "Уйдите слоном с e4, чтобы ладья e1 объявила шах королю e8.",
                "fen": "1n2k3/p7/5r2/P5p1/Npp1B1P1/6P1/P7/4R1K1 w - - 0 25",
                "source_square": "e4",
                "difficulty": 2,
                "payload": {
                    "piece": "B",
                    "mode": "accepted_moves",
                    "accepted_moves": ["e4d5"],
                },
                "explanation": "Bd5 открывает вертикаль e, и ладья e1 объявляет шах королю e8.",
                "sort_order": 1,
            },
            {
                "task_type": "choose_option",
                "title": "Чем вскрытый шах сильнее обычного",
                "instruction": "Выберите, чем вскрытый шах сильнее обычного.",
                "fen": "1n2k3/p7/5r2/P5p1/Npp1B1P1/6P1/P7/4R1K1 w - - 0 25",
                "difficulty": 2,
                "payload": {
                    "options": _options(
                        ("free_move", "Уходящая фигура может одновременно создать вторую угрозу"),
                        ("no_reply", "Соперник не имеет права отвечать на шах"),
                        ("mate", "Вскрытый шах всегда ставит мат"),
                        ("capture", "Он всегда выигрывает ферзя"),
                    ),
                    "correct_option": "free_move",
                },
                "explanation": "Пока соперник отвечает на шах, уходящая фигура успевает нанести второй удар.",
                "sort_order": 2,
            },
        ],
    },
    {
        "slug": "combination",
        "title": "Что такое комбинация",
        "section_title": "Комбинация",
        "theory": (
            "Комбинация — это форсированная последовательность ходов с жертвой или "
            "неочевидным ударом, которая приводит к выигрышу материала или мату. "
            "Комбинацию объединяют конкретная цель и точный расчёт вариантов."
        ),
        "sort_order": 8,
        "tasks": [
            {
                "task_type": "make_move",
                "title": "Мат ферзём на f7",
                "instruction": "Поставьте мат в один ход: ферзь f3, слон c4 уже готовы помочь.",
                "fen": "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
                "source_square": "f3",
                "difficulty": 2,
                "payload": {
                    "piece": "Q",
                    "mode": "accepted_moves",
                    "accepted_moves": ["f3f7"],
                },
                "explanation": "Qxf7# — ферзя защищает слон c4, поэтому король e8 не может его забрать.",
                "sort_order": 1,
            },
            {
                "task_type": "choose_option",
                "title": "Что делает комбинацию комбинацией",
                "instruction": "Выберите, что отличает комбинацию.",
                "fen": "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
                "difficulty": 2,
                "payload": {
                    "options": _options(
                        ("forcing", "Форсированная последовательность с конкретной целью"),
                        ("random", "Любая серия ходов на удачу"),
                        ("trades", "Только размены фигур"),
                        ("quiet", "Спокойный позиционный ход"),
                    ),
                    "correct_option": "forcing",
                },
                "explanation": "Комбинация всегда имеет точный расчёт и ясную цель — мат или выигрыш материала.",
                "sort_order": 2,
            },
        ],
    },
]


MIDDLEGAME_LESSONS = [
    {
        "slug": "material-advantage",
        "title": "Материальное преимущество",
        "section_title": "Материальное преимущество",
        "theory": (
            "Материальное преимущество — это лишние фигуры или пешки. Его проще всего "
            "использовать, разменивая фигуры: чем меньше фигур на доске, тем труднее "
            "сопернику создать контригру."
        ),
        "sort_order": 1,
        "tasks": [
            {
                "task_type": "make_move",
                "title": "Заберите лишнюю ладью",
                "instruction": "Ладья d4 ничем не защищена. Заберите её ладьёй d1.",
                "fen": "4k1N1/1n6/ppp3p1/4P1b1/3r4/5R2/PP5P/3RK3 w - - 0 37",
                "source_square": "d1",
                "difficulty": 1,
                "payload": {
                    "piece": "R",
                    "mode": "accepted_moves",
                    "accepted_moves": ["d1d4"],
                },
                "explanation": "Rxd4 выигрывает ладью и увеличивает материальный перевес.",
                "sort_order": 1,
            },
            {
                "task_type": "choose_option",
                "title": "Как использовать лишний материал",
                "instruction": "Выберите лучший способ использовать лишний материал.",
                "fen": "4k1N1/1n6/ppp3p1/4P1b1/3r4/5R2/PP5P/3RK3 w - - 0 37",
                "difficulty": 2,
                "payload": {
                    "options": _options(
                        ("trade", "Разменивать фигуры и упрощать позицию"),
                        ("attack", "Без подготовки идти на рискованную атаку"),
                        ("give", "Отдавать фигуры за инициативу"),
                        ("wait", "Ничего не делать и повторять ходы"),
                    ),
                    "correct_option": "trade",
                },
                "explanation": "При лишнем материале размены приближают победу: меньше фигур — меньше шансов у соперника.",
                "sort_order": 2,
            },
        ],
    },
    {
        "slug": "positional-advantage",
        "title": "Позиционное преимущество",
        "section_title": "Позиционное преимущество",
        "theory": (
            "Позиционное преимущество — это лучшая позиция фигур, контроль над полями "
            "и линиями без немедленного выигрыша материала. Его наращивают спокойными "
            "улучшающими ходами."
        ),
        "sort_order": 2,
        "tasks": [
            {
                "task_type": "choose_option",
                "title": "Что такое позиционное преимущество",
                "instruction": "Выберите определение позиционного преимущества.",
                "fen": "4k2n/p6p/5R2/p1P5/2bN2Pp/8/1P4BP/N3K3 w - - 0 27",
                "difficulty": 1,
                "payload": {
                    "options": _options(
                        ("position", "Лучшее расположение фигур и контроль над полями"),
                        ("material", "Только лишние фигуры"),
                        ("clock", "Преимущество во времени"),
                        ("luck", "Случайная удача"),
                    ),
                    "correct_option": "position",
                },
                "explanation": "Позиционное преимущество не измеряется материалом — оно в качестве позиции.",
                "sort_order": 1,
            },
            {
                "task_type": "make_move",
                "title": "Улучшите коня",
                "instruction": "Переведите коня a1 ближе к центру на c2.",
                "fen": "4k2n/p6p/5R2/p1P5/2bN2Pp/8/1P4BP/N3K3 w - - 0 27",
                "source_square": "a1",
                "difficulty": 1,
                "payload": {
                    "piece": "N",
                    "mode": "accepted_moves",
                    "accepted_moves": ["a1c2"],
                },
                "explanation": "Nc2 — улучшающий ход: конь ближе к центру и активнее.",
                "sort_order": 2,
            },
        ],
    },
    {
        "slug": "pawn-structure",
        "title": "Расположение пешек",
        "section_title": "Расположение пешек",
        "theory": (
            "Пешечная структура определяет слабые и сильные поля. Сдвоенные и изолированные "
            "пешки часто становятся мишенями, а пешечные островки мешают защите. "
            "Здоровые пешки — опора для фигур."
        ),
        "sort_order": 3,
        "tasks": [
            {
                "task_type": "choose_option",
                "title": "Чем плохи сдвоенные пешки",
                "instruction": "Выберите, почему сдвоенные пешки считаются слабостью.",
                "fen": "4k3/2q5/8/1p1P1Ppp/8/P4P2/n2P4/b3K1BN w - - 0 26",
                "difficulty": 1,
                "payload": {
                    "options": _options(
                        ("immobile", "Они менее подвижны и не могут защищать друг друга"),
                        ("strong", "Они всегда сильнее обычных"),
                        ("center", "Они захватывают центр"),
                        ("promote", "Они быстрее проходят в ферзи"),
                    ),
                    "correct_option": "immobile",
                },
                "explanation": "Сдвоенные пешки стесняют друг друга и часто становятся слабостью.",
                "sort_order": 1,
            },
            {
                "task_type": "make_move",
                "title": "Захватите центр пешкой",
                "instruction": "Продвиньте пешку d2 на два поля вперёд.",
                "fen": "4k3/2q5/8/1p1P1Ppp/8/P4P2/n2P4/b3K1BN w - - 0 26",
                "source_square": "d2",
                "difficulty": 1,
                "payload": {
                    "piece": "P",
                    "mode": "accepted_moves",
                    "accepted_moves": ["d2d4"],
                },
                "explanation": "d2-d4 занимает центр и даёт пешке больше пространства.",
                "sort_order": 2,
            },
        ],
    },
    {
        "slug": "open-files",
        "title": "Открытые линии",
        "section_title": "Открытые линии",
        "theory": (
            "Открытая линия — вертикаль без пешек. Ладьи и ферзи на такой линии "
            "проникают в лагерь соперника и атакуют слабые пункты. Занимать открытые "
            "линии нужно первыми."
        ),
        "sort_order": 4,
        "tasks": [
            {
                "task_type": "choose_option",
                "title": "Что такое открытая линия",
                "instruction": "Выберите определение открытой линии.",
                "fen": "4k1N1/7p/1r6/4Pnp1/8/3P1b1P/2R5/R3K3 w - - 0 30",
                "difficulty": 1,
                "payload": {
                    "options": _options(
                        ("no_pawns", "Вертикаль, на которой нет пешек"),
                        ("no_pieces", "Горизонталь без фигур"),
                        ("diagonal", "Диагональ без фигур"),
                        ("rank", "Восьмая горизонталь"),
                    ),
                    "correct_option": "no_pawns",
                },
                "explanation": "Открытой называют вертикаль, свободную от пешек.",
                "sort_order": 1,
            },
            {
                "task_type": "make_move",
                "title": "Займите открытую линию",
                "instruction": "Переведите ладью с a1 на открытую вертикаль d.",
                "fen": "4k1N1/7p/1r6/4Pnp1/8/3P1b1P/2R5/R3K3 w - - 0 30",
                "source_square": "a1",
                "difficulty": 2,
                "payload": {
                    "piece": "R",
                    "mode": "accepted_moves",
                    "accepted_moves": ["a1d1"],
                },
                "explanation": "Rd1 занимает открытую вертикаль d и может проникать в лагерь соперника.",
                "sort_order": 2,
            },
        ],
    },
    {
        "slug": "good-bad-pieces",
        "title": "Хорошие и плохие фигуры",
        "section_title": "Хорошие и плохие фигуры",
        "theory": (
            "Хорошая фигура активна, контролирует важные поля и имеет много ходов. "
            "Плохая фигура стоит на краю, заперта своими пешками или не участвует "
            "в игре. Улучшать нужно в первую очередь плохие фигуры."
        ),
        "sort_order": 5,
        "tasks": [
            {
                "task_type": "choose_option",
                "title": "Какая фигура обычно плохая",
                "instruction": "Выберите, какая фигура считается плохой.",
                "fen": "4k2n/p6p/5R2/p1P5/2bN2Pp/8/1P4BP/N3K3 w - - 0 27",
                "difficulty": 1,
                "payload": {
                    "options": _options(
                        ("rim", "Конь на краю доски с малым числом ходов"),
                        ("center", "Конь в центре доски"),
                        ("active", "Активная ладья на открытой линии"),
                        ("bishop", "Слон на длинной диагонали"),
                    ),
                    "correct_option": "rim",
                },
                "explanation": "«Конь на краю — горе в семье»: у него меньше полей и он менее активен.",
                "sort_order": 1,
            },
            {
                "task_type": "make_move",
                "title": "Выведите коня с края",
                "instruction": "Переведите коня a1 на b3, ближе к центру.",
                "fen": "4k2n/p6p/5R2/p1P5/2bN2Pp/8/1P4BP/N3K3 w - - 0 27",
                "source_square": "a1",
                "difficulty": 1,
                "payload": {
                    "piece": "N",
                    "mode": "accepted_moves",
                    "accepted_moves": ["a1b3"],
                },
                "explanation": "Nb3 улучшает фигуру: конь уходит с края и берёт под контроль центральные поля.",
                "sort_order": 2,
            },
        ],
    },
    {
        "slug": "middlegame-mistakes",
        "title": "Типовые ошибки в миттельшпиле",
        "section_title": "Типовые ошибки",
        "theory": (
            "Частые ошибки миттельшпиля: ранний вывод ферзя под темпы, охота за "
            "пешками в ущерб развитию, недооценка открытых линий и слабостей "
            "собственного короля. Перед ходом полезно проверить, не создаёт ли он "
            "новую слабость."
        ),
        "sort_order": 6,
        "tasks": [
            {
                "task_type": "choose_option",
                "title": "Какая ошибка типична для миттельшпиля",
                "instruction": "Выберите типичную ошибку миттельшпиля.",
                "fen": "4k1N1/7p/1r6/4Pnp1/8/3P1b1P/2R5/R3K3 w - - 0 30",
                "difficulty": 2,
                "payload": {
                    "options": _options(
                        ("queen_early", "Ранний вывод ферзя под удары и темпы"),
                        ("develop", "Развитие фигур к центру"),
                        ("castle", "Своевременная рокировка"),
                        ("trade", "Размен при лишнем материале"),
                    ),
                    "correct_option": "queen_early",
                },
                "explanation": "Ферзь — ценная фигура: выводить его рано опасно, соперник выигрывает темпы.",
                "sort_order": 1,
            },
            {
                "task_type": "choose_option",
                "title": "Что проверить перед ходом",
                "instruction": "Выберите, что нужно проверить перед ходом.",
                "fen": "4k1N1/7p/1r6/4Pnp1/8/3P1b1P/2R5/R3K3 w - - 0 30",
                "difficulty": 2,
                "payload": {
                    "options": _options(
                        ("weakness", "Не создаёт ли ход новую слабость или открытую линию"),
                        ("speed", "Только скорость хода"),
                        ("color", "Цвет фигур"),
                        ("notation", "Правильность записи хода"),
                    ),
                    "correct_option": "weakness",
                },
                "explanation": "Главная профилактика ошибок — проверка слабостей и открытых линий после хода.",
                "sort_order": 2,
            },
        ],
    },
]


LESSONS_BY_MODULE = {
    "pieces": PIECES_LESSONS,
    "king-safety": KING_SAFETY_LESSONS,
    "tactics": TACTICS_LESSONS,
    "endgames": ENDGAMES_LESSONS,
    "openings": OPENINGS_LESSONS,
    "middlegame": MIDDLEGAME_LESSONS,
}
