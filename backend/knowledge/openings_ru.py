"""Перевод названий дебютов на русский язык.

База знаний ``backend/knowledge/openings.json`` содержит 3810 записей
с английскими именами (3174 уникальных), поля ``name_ru`` нет. Названия
дебютов структурированы::

    Sicilian Defense: Najdorf Variation, English Attack
    ^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^
    семья              подвариант        линия

Поэтому перевод собран из нескольких уровней, от точного к общему:

1. ``FAMILIES``    — 141 базовая семья («Sicilian Defense» → «Сицилианская
   защита»). Словарь конечен, переведён вручную.
2. ``COMPONENTS``  — частые подварианты целиком («Main Line» → «Основная
   линия»), включая составные без типа («Old Benoni»).
3. ``DESCRIPTORS`` — прилагательные-головы («French» → «Французский»).
   Согласуются в роде с типом дебюта: «French Defense» → «Французская
   защита», «French Variation» → «Французский вариант».
4. ``EONYMS``      — имена игроков в родительном падеже («Najdorf» →
   «Найдорфа»): «вариант Найдорфа», «защита Морфи».
5. Транслитерация — для неизвестных имён («Krause» → «вариант Краусе»).
   Это осознанный фолбэк: лучше родное по звучанию «вариант Краусе», чем
   английское Krause Variation. Список имён бесконечен (1387 голов), так
   что полный словарь вручную нереально поддерживать.
6. ``with <ход>`` — ходы переводятся буквой фигуры: «with Nf3» → «с Kf3».

Если хоть один компонент перевести не удалось, функция возвращает
``None`` — потребитель показывает оригинальное английское имя. Покрытие
расширяется досыпкой словарей, остальной код не меняется.
"""

from __future__ import annotations

import re

# Разбиваем имя на компоненты, сохраняя разделители ": " и ", ".
_SPLIT_RE = re.compile(r"(:\s*|,\s*)")

# Типы дебюта и их род (для согласования прилагательных).
_SUFFIX_KINDS = {
    "Variation": "вариант",
    "Line": "линия",
    "Defense": "защита",
    "Gambit": "гамбит",
    "Countergambit": "контргамбит",
    "Attack": "атака",
    "System": "система",
    "Opening": "начало",
    "Game": "партия",
    "Formation": "формация",
    "Bind": "стяжение",
    "Order": "порядок",
    "Counterattack": "контратака",
    "Counter-Gambit": "контргамбит",
}

_KIND_GENDER = {
    "вариант": "m",
    "гамбит": "m",
    "контргамбит": "m",
    "порядок": "m",
    "защита": "f",
    "атака": "f",
    "контратака": "f",
    "система": "f",
    "партия": "f",
    "линия": "f",
    "формация": "f",
    "стяжение": "n",
    "начало": "n",
}

# «Accepted/Declined» — в русской традиции это пояснение в скобках.
_ACCEPTED_DECLINED = {
    "Accepted": "принятый",
    "Declined": "отклонённый",
}

# «with Nf3», «with e3», «with Bc4» — ход или продвижение.
_WITH_RE = re.compile(r"^with\s+(.+)$")
_MOVE_CHARS = re.compile(r"^\.?[KQRBNabcdefgh1-8x ]+$")

# Тип дебюта в творительном падеже — нужно после предлога «с»:
# «with Max Lange Defense» → «с защитой Макса Ланге».
_KIND_INSTRUMENTAL = {
    "вариант": "вариантом",
    "гамбит": "гамбитом",
    "контргамбит": "контргамбитом",
    "порядок": "порядком",
    "защита": "защитой",
    "атака": "атакой",
    "контратака": "контратакой",
    "система": "системой",
    "партия": "партией",
    "линия": "линией",
    "формация": "формацией",
    "стяжение": "стяжением",
    "начало": "началом",
}

# Буквы фигур в русской нотации: конь, слон, ладья, ферзь, король.
_PIECE_LETTERS = {"N": "К", "B": "С", "R": "Л", "Q": "Ф", "K": "К"}

# Слова-префиксы, отрезаемые от прилагательной головы:
# «Anglo-Scandinavian Defense» → «англоскандинавская защита».
# Работают только когда хвост — прилагательное из DESCRIPTORS.
_PREFIXES: dict[str, str] = {
    "Neo-": "нео",
    "Neo ": "нео",
    "Anti-": "анти",
    "Anti ": "анти",
    "Pseudo-": "псевдо",
    "Semi-": "полу",
    "Hyper-": "гипер",
    "Ultra-": "ультра",
    "Anglo-": "англо",
    "Franco-": "франко",
    "Double ": "двойной ",
    "Reversed ": "обращённый ",
    "Delayed ": "отложенный ",
    "Accelerated ": "ускоренный ",
    "Early ": "ранний ",
    "Old ": "старый ",
    "New ": "новый ",
}


# ---------------------------------------------------------------------------
# Базовые семьи дебютов — все 141 имя из openings.json.
# ---------------------------------------------------------------------------

FAMILIES: dict[str, str] = {
    "Alekhine Defense": "Защита Алехина",
    "Amar Opening": "Начало Амара",
    "Amazon Attack": "Атака амазонки",
    "Amsterdam Attack": "Амстердамская атака",
    "Anderssen's Opening": "Начало Андерсена",
    "Australian Defense": "Австралийская защита",
    "Barnes Defense": "Защита Барнса",
    "Barnes Opening": "Начало Барнса",
    "Basque Opening": "Баскское начало",
    "Benko Gambit": "Гамбит Бенко",
    "Benko Gambit Accepted": "Гамбит Бенко (принятый)",
    "Benko Gambit Declined": "Гамбит Бенко (отклонённый)",
    "Benoni Defense": "Защита Бенони",
    "Bird Opening": "Начало Бёрда",
    "Bishop's Opening": "Начало слона",
    "Blackmar-Diemer Gambit": "Гамбит Блэкмар — Димера",
    "Blackmar-Diemer Gambit Accepted": "Гамбит Блэкмар — Димера (принятый)",
    "Blackmar-Diemer Gambit Declined": "Гамбит Блэкмар — Димера (отклонённый)",
    "Blumenfeld Countergambit": "Контргамбит Блюменфельда",
    "Blumenfeld Countergambit Accepted": "Контргамбит Блюменфельда (принятый)",
    "Bogo-Indian Defense": "Бого-индийская защита",
    "Bongcloud Attack": "Атака «Бонгклауд»",
    "Borg Defense": "Защита Борга",
    "Canard Opening": "Начало Канара",
    "Caro-Kann Defense": "Защита Каро — Канна",
    "Carr Defense": "Защита Карра",
    "Catalan Opening": "Каталанское начало",
    "Center Game": "Центральная партия",
    "Center Game Accepted": "Центральная партия (принятая)",
    "Clemenz Opening": "Начало Клементса",
    "Colle System": "Система Колле",
    "Creepy Crawly Formation": "Ползучая формация",
    "Czech Defense": "Чешская защита",
    "Danish Gambit": "Датский гамбит",
    "Danish Gambit Accepted": "Датский гамбит (принятый)",
    "Danish Gambit Declined": "Датский гамбит (отклонённый)",
    "Dresden Opening": "Дрезденское начало",
    "Duras Gambit": "Гамбит Дураса",
    "Dutch Defense": "Голландская защита",
    "Döry Defense": "Защита Дёри",
    "East Indian Defense": "Восточно-индийская защита",
    "Elephant Gambit": "Слоновий гамбит",
    "English Defense": "Английская защита",
    "English Opening": "Английское начало",
    "English Orangutan": "Английский орангутан",
    "Englund Gambit": "Гамбит Энглунда",
    "Englund Gambit Declined": "Гамбит Энглунда (отклонённый)",
    "Formation": "Формация",
    "Four Knights Game": "Партия четырёх коней",
    "French Defense": "Французская защита",
    "Fried Fox Defense": "Защита «жареная лиса»",
    "Global Opening": "Всемирное начало",
    "Goldsmith Defense": "Защита Голдсмита",
    "Grob Opening": "Начало Гроба",
    "Grünfeld Defense": "Защита Грюнфельда",
    "Gunderam Defense": "Защита Гундерама",
    "Hippopotamus Defense": "Защита «бегемот»",
    "Horwitz Defense": "Защита Горвица",
    "Hungarian Opening": "Венгерское начало",
    "Indian Defense": "Индийская защита",
    "Irish Gambit": "Ирландский гамбит",
    "Italian Game": "Итальянская партия",
    "Kangaroo Defense": "Защита «кенгуру»",
    "King's Gambit": "Королевский гамбит",
    "King's Gambit Accepted": "Королевский гамбит (принятый)",
    "King's Gambit Declined": "Королевский гамбит (отклонённый)",
    "King's Indian Attack": "Индийское нападение",
    "King's Indian Defense": "Индийская защита",
    "King's Knight Opening": "Начало королевского коня",
    "King's Pawn Game": "Партия королевской пешки",
    "King's Pawn Opening": "Начало королевской пешки",
    "Kádas Opening": "Начало Кадаша",
    "Lasker Simul Special": "Специальный вариант Ласкера",
    "Latvian Gambit": "Латышский гамбит",
    "Latvian Gambit Accepted": "Латышский гамбит (принятый)",
    "Lemming Defense": "Защита леммингов",
    "Lion Defense": "Львиная защита",
    "London System": "Лондонская система",
    "Marienbad System": "Система Мариенбад",
    "Mexican Defense": "Мексиканская защита",
    "Mieses Opening": "Начало Мизеса",
    "Mikenas Defense": "Защита Микенаса",
    "Modern Defense": "Современная защита",
    "Montevideo Defense": "Защита Монтевидео",
    "Neo-Grünfeld Defense": "Защита нео-Грюнфельда",
    "Nimzo-Indian Defense": "Защита Нимцовича",
    "Nimzo-Larsen Attack": "Атака Нимцовича — Ларсена",
    "Nimzowitsch Defense": "Защита Нимцовича",
    "Old Indian Defense": "Староиндийская защита",
    "Owen Defense": "Защита Оуэна",
    "Paleface Attack": "Атака бледного лица",
    "Petrov's Defense": "Защита Петрова",
    "Philidor Defense": "Защита Филидора",
    "Pirc Defense": "Защита Перца",
    "Polish Defense": "Польская защита",
    "Polish Opening": "Польское начало",
    "Ponziani Opening": "Начало Понциани",
    "Portuguese Opening": "Португальское начало",
    "Pseudo Queen's Indian Defense": "Ложно-ферзево-индийская защита",
    "Pterodactyl Defense": "Защита птеродактиля",
    "Queen's Gambit": "Ферзевый гамбит",
    "Queen's Gambit Accepted": "Ферзевый гамбит (принятый)",
    "Queen's Gambit Declined": "Ферзевый гамбит (отклонённый)",
    "Queen's Indian Accelerated": "Ускоренная ферзево-индийская защита",
    "Queen's Indian Defense": "Ферзево-индийская защита",
    "Queen's Pawn": "Ферзевая пешка",
    "Queen's Pawn Game": "Партия ферзевой пешки",
    "Rapport-Jobava System": "Система Раппорта — Джобавы",
    "Rat Defense": "Защита «крыса»",
    "Richter-Veresov Attack": "Атака Рихтера — Вересова",
    "Robatsch Defense": "Защита Робача",
    "Rubinstein Opening": "Начало Рубинштейна",
    "Ruy Lopez": "Испанская партия",
    "Réti Opening": "Начало Рети",
    "Saragossa Opening": "Начало Сарагоссы",
    "Scandinavian Defense": "Скандинавская защита",
    "Scotch Game": "Шотландская партия",
    "Semi-Slav Defense": "Полуславянская защита",
    "Semi-Slav Defense Accepted": "Полуславянская защита (принятая)",
    "Sicilian Defense": "Сицилианская защита",
    "Slav Defense": "Славянская защита",
    "Slav Indian": "Славяно-индийская защита",
    "Sodium Attack": "Атака натрия",
    "St. George Defense": "Защита Святого Георга",
    "Tarrasch Defense": "Защита Тарраша",
    "Three Knights Opening": "Начало трёх коней",
    "Torre Attack": "Атака Торре",
    "Trompowsky Attack": "Атака Тромповского",
    "Valencia Opening": "Начало Валенсии",
    "Van Geet Opening": "Начало ван Гета",
    "Van't Kruijs Opening": "Начало ван Крейса",
    "Vienna Gambit": "Венский гамбит",
    "Vienna Game": "Венская партия",
    "Vulture Defense": "Защита грифа",
    "Wade Defense": "Защита Уэйда",
    "Ware Opening": "Начало Уэра",
    "Ware Defense": "Защита Уэра",
    "Yusupov-Rubinstein System": "Система Юсупова — Рубинштейна",
    "Zaire Defense": "Защита Заира",
    "Zukertort Opening": "Начало Цукерторта",
    "Zukertort Defense": "Защита Цукерторта",
}


# ---------------------------------------------------------------------------
# Подварианты, которые переводятся целиком (имеют нестандартную форму).
# ---------------------------------------------------------------------------

COMPONENTS: dict[str, str] = {
    "Accelerated Dragon": "Ускоренный Дракон",
    "Accelerated London System": "Ускоренная лондонская система",
    "Accelerated Panov Attack": "Ускоренная атака Панова",
    "Agincourt Defense": "Защита Азенкура",
    "Alekhine-Chatard Attack": "Атака Алехина — Шатара",
    "Albin Countergambit": "Контргамбит Альбина",
    "Anglo-Grünfeld Variation": "Англогрюнфельдский вариант",
    "Anglo-Indian Defense": "Англо-индийская защита",
    "Anglo-Slav Variation": "Англо-славянский вариант",
    "Anti-Berlin Variation": "Антиберлинский вариант",
    "Bayonet Attack": "Штыковая атака",
    "Berger Variation": "вариант Бергера",
    "Bernstein Variation": "вариант Бернштейна",
    "Bishop's Gambit": "Гамбит слона",
    "Bishop's Variation": "Слоновый вариант",
    "Bogoljubow Defense": "защита Боголюбова",
    "Bogoljubow Variation": "вариант Боголюбова",
    "Boleslavsky Variation": "вариант Болеславского",
    "Botvinnik System Reversed": "Обращённая система Ботвинника",
    "Botvinnik Variation": "вариант Ботвинника",
    "Breyer Variation": "вариант Брейера",
    "Brinckmann Attack": "Атака Бринкманна",
    "Bronstein Variation": "вариант Бронштейна",
    "Budapest Gambit Accepted": "Будапештский гамбит (принятый)",
    "Cambridge Springs Defense": "Кембридж-спрингская защита",
    "Carlsbad Variation": "вариант Карлсбада",
    "Capablanca Variation": "вариант Капабланки",
    "Central": "Центральная",
    "Central Variation": "Центральный вариант",
    "Central Line": "Центральная линия",
    "Charousek Gambit": "Гамбит Чарошека",
    "Chigorin Defense": "защита Чигорина",
    "Chigorin Variation": "вариант Чигорина",
    "Classical": "Классическая",
    "Classical Attack": "Классическая атака",
    "Classical Defense": "Классическая защита",
    "Classical Line": "Классическая линия",
    "Classical Variation": "Классический вариант",
    "Closed": "Закрытая",
    "Closed Variation": "Закрытый вариант",
    "Colle System": "Система Колле",
    "Delayed Alapin Variation": "Отложенный вариант Алапина",
    "Delayed Exchange Variation": "Отложенный разменный вариант",
    "Dragon Variation": "вариант Дракона",
    "Eastern": "Восточная",
    "Endgame Variation": "Эндшпильный вариант",
    "Exchange Variation": "Разменный вариант",
    "Evans Gambit Declined": "Гамбит Эванса (отклонённый)",
    "Evans Gambit": "Гамбит Эванса",
    "Falkbeer Countergambit": "Контргамбит Фалькбера",
    "Fianchetto": "Фианхетто",
    "Fianchetto Defense": "Защита с фианхетто",
    "Fianchetto Line": "Линия с фианхетто",
    "Fianchetto Variation": "Вариант с фианхетто",
    "Flohr Variation": "вариант Флора",
    "Four Knights Variation": "Вариант четырёх коней",
    "Four Knights Attack": "Атака четырёх коней",
    "Four Pawns Attack": "Атака четырёх пешек",
    "From's Gambit": "Гамбит Фрома",
    "Gligoric System": "Система Глигорича",
    "Giuoco Pianissimo": "Джиоко Пианиссимо",
    "Giuoco Piano": "Джиоко Пиано",
    "Greco Gambit": "Гамбит Греве",
    "Grob Gambit": "Гамбит Гроба",
    "Gurevich System": "Система Гуревича",
    "Hanham Variation": "вариант Хэмхема",
    "Harrwitz Attack": "Атака Харвица",
    "Henneberger Variation": "вариант Хеннебергера",
    "Ilyin-Zhenevsky Variation": "вариант Ильина — Женевского",
    "Jaenisch Counterattack": "Контратака Яниша",
    "Janowski Variation": "вариант Яновского",
    "Kan Variation": "вариант Кана",
    "Karpov Variation": "вариант Карпова",
    "Kasparov-Petrosian Variation": "вариант Каспарова — Петросяна",
    "Kazan Variation": "вариант Казани",
    "Kennedy Variation": "вариант Кеннеди",
    "Keres Variation": "вариант Кереса",
    "Kieseritzky Gambit": "Гамбит Кисерицкого",
    "King's English Variation": "Королевский английский вариант",
    "King's Knight's Gambit": "Гамбит королевского коня",
    "King's Pawn Line": "Линия королевской пешки",
    "Krause Variation": "вариант Краусе",
    "Lasker Defense": "защита Ласкера",
    "Lasker-Pelikan Variation": "вариант Ласкера — Пеликана",
    "Lasker Variation": "вариант Ласкера",
    "Levenfish Variation": "вариант Левенфиша",
    "Lemberger Countergambit": "Контргамбит Лембергера",
    "Leningrad Variation": "Ленинградский вариант",
    "Leonhardt Variation": "вариант Леонгарта",
    "Lion Variation": "Львиный вариант",
    "London Defensive System": "Лондонская оборонительная система",
    "London System": "Лондонская система",
    "Main Line": "Основная линия",
    "Maroczy Bind": "Стяжение Мароци",
    "Maroczy Variation": "вариант Мароци",
    "Marshall Attack": "Атака Маршалла",
    "Marshall Gambit": "Гамбит Маршалла",
    "Max Lange Attack": "Атака Макса Ланге",
    "McCutcheon Variation": "вариант МакКатчона",
    "McDonnell Attack": "Атака Мак-Доннелла",
    "Meran Variation": "вариант Мерана",
    "Milner-Barry Gambit": "Гамбит Милнер — Барри",
    "Mikenas-Carls Variation": "вариант Микенаса — Карлса",
    "Modern": "Современная",
    "Modern Attack": "Современная атака",
    "Modern Line": "Современная линия",
    "Modern Steinitz Defense": "Современная защита Штейница",
    "Modern Variation": "Современный вариант",
    "Modern Variations": "Современные варианты",
    "Morphy Defense": "защита Морфи",
    "Moscow Variation": "Московский вариант",
    "Najdorf Variation": "вариант Найдорфа",
    "Neo-Catalan Declined": "Неокаталанское начало (отклонённое)",
    "Nimzowitsch Variation": "вариант Нимцовича",
    "Noa Variation": "вариант Ноа",
    "Normal Line": "Нормальная линия",
    "Normal System": "Нормальная система",
    "Normal Variation": "Нормальный вариант",
    "Old Benoni": "Старый Бенони",
    "Old Variation": "Старый вариант",
    "Opocensky Variation": "вариант Опоценского",
    "Open Defense": "Открытая защита",
    "Open Line": "Открытая линия",
    "Open System": "Открытая система",
    "Open": "Открытая",
    "Orthodox Defense": "Ортодоксальная защита",
    "Orthodox Line": "Ортодоксальная линия",
    "Orthodox Variation": "Ортодоксальный вариант",
    "O'Kelly Variation": "вариант О'Келли",
    "Panov Attack": "Атака Панова",
    "Panteldakis Countergambit": "Контргамбит Пантелдакиса",
    "Paulsen Variation": "вариант Паульсена",
    "Petrosian Variation": "вариант Петросяна",
    "Polerio Defense": "защита Полерио",
    "Poisoned Pawn Variation": "Вариант с отравленной пешкой",
    "Portuguese Gambit": "Португальский гамбит",
    "Positional Variation": "Позиционный вариант",
    "Pterodactyl Variation": "Вариант птеродактиля",
    "Quiet Line": "Тихая линия",
    "Quiet System": "Тихая система",
    "Quiet Variation": "Тихий вариант",
    "Reshevsky Variation": "вариант Решевского",
    "Retreat Variation": "Вариант с отступлением",
    "Reversed Closed Sicilian": "Обращённая закрытая сицилианская",
    "Richter-Rauzer Variation": "вариант Рихтера — Раузера",
    "Romanishin Variation": "вариант Романишина",
    "Rubinstein System": "Система Рубинштейна",
    "Rubinstein Variation": "вариант Рубинштейна",
    "Russian Variation": "Русский вариант",
    "Salvio Gambit": "Гамбит Сальвио",
    "Scandinavian Variation": "Скандинавский вариант",
    "Schlechter Variation": "вариант Шлехтера",
    "Scotch Gambit": "Шотландский гамбит",
    "Scotch Variation": "Шотландский вариант",
    "Schliemann Defense": "Защита Шлиеманна",
    "Scheveningen Variation": "вариант Схевенингена",
    "Semi-Tarrasch Defense": "Полутаррашева защита",
    "Sicilian": "Сицилианская",
    "Smyslov Variation": "вариант Смыслова",
    "Smith-Morra Gambit Accepted": "Гамбит Смита — Морры (принятый)",
    "Smith-Morra Gambit Declined": "Гамбит Смита — Морры (отклонённый)",
    "Sokolsky Variation": "вариант Сокольского",
    "Spassky Variation": "вариант Спасского",
    "Spielmann Variation": "вариант Шпильманна",
    "Sveshnikov Variation": "вариант Свешникова",
    "Staunton Gambit": "Гамбит Стонтона",
    "Steinitz Variation": "вариант Штейница",
    "Steiner Attack": "Атака Штейнера",
    "Steiner Variation": "вариант Штейнера",
    "Stoltz Variation": "вариант Штольца",
    "St. Petersburg Variation": "Петербургский вариант",
    "Stonewall Variation": "Вариант «Каменная стена»",
    "Symmetrical Variation": "Симметричный вариант",
    "Taimanov Variation": "вариант Тайманова",
    "Tarrasch Defense": "защита Тарраша",
    "Tarrasch Variation": "вариант Тарраша",
    "Tartakower Variation": "вариант Тартаковера",
    "Three Knights Variation": "Вариант трёх коней",
    "Three Knights": "Трёхконевая",
    "Torre Attack": "Атака Торре",
    "Traditional Line": "Традиционная линия",
    "Traditional System": "Традиционная система",
    "Traditional Variation": "Традиционный вариант",
    "Two Knights Attack": "Атака двух коней",
    "Two Knights Defense": "Защита двух коней",
    "Two Knights Variation": "Вариант двух коней",
    "Venice System": "Венецианская система",
    "Vienna Gambit": "Венский гамбит",
    "Wing Attack": "Крыльевая атака",
    "Wing Gambit": "Крыльевой гамбит",
    "Winawer Variation": "вариант Винавера",
    "Yugoslav Attack": "Югославская атака",
    "Yugoslav Variation": "Югославский вариант",
    "Zilbermints Gambit": "Гамбит Зильберминца",
    # --- остатки базы: редкие варианты, ловушки, гибриды ---
    "150 Attack": "Атака 150",
    "Accelerated Averbakh Variation": "Ускоренный вариант Авербаха",
    "Accelerated Gurgenidze": "Ускоренный Гургенидзе",
    "Accelerated Mannheim Variation": "Ускоренный вариант Мангейма",
    "Accelerated Meran Variation": "Ускоренный вариант Мерана",
    "Accelerated Move Order": "Ускоренный порядок ходов",
    "Acton Extension": "Расширение Эктона",
    "Adams' Gambit": "Гамбит Адамса",
    "Advance": "Продвижение",
    "Allgaier": "Алльгайер",
    "Anglo-Lithuanian Variation": "Англо-литовский вариант",
    "Anti-Alapin Gambit": "Анти-гамбит Алапина",
    "Anti-Benoni Variation": "Анти-вариант Бенони",
    "Anti-Cambridge Springs Variation": "Анти-вариант Кембридж-Спрингс",
    "Anti-Fried Liver Defense": "Анти-защита «жареная печень»",
    "Anti-Moscow Gambit": "Анти-московский гамбит",
    "Anti-Noteboom Gambit": "Анти-гамбит Нотбума",
    "Anti-Noteboom Variation": "Анти-вариант Нотбума",
    "Anti-Qxd4 Move Order": "Анти-порядок ходов с Фхd4",
    "Anti-Qxd4 Move Order Accepted": "Анти-порядок ходов с Фхd4 (принятый)",
    "Anti-Sozin Variation": "Анти-вариант Созина",
    "Anti-Sveshnikov Variation": "Анти-вариант Свешникова",
    "Anti-Tartakower Variation": "Анти-вариант Тартаковера",
    "Anti-Winawer Gambit": "Анти-гамбит Винавера",
    "Austriadactylus Western": "Австриадактиль (западный)",
    "Batavo-Polish Attack": "Батавско-польская атака",
    "Beginner's Trap": "Ловушка новичка",
    "Benoni": "Бенони",
    "Benoni Beefeater Pterodactyl": "Птеродактиль Бенони-Бифеатер",
    "Benoni Pteranodon": "Бенони-птеранодон",
    "Benoni Pterodactyl": "Бенони-птеродактиль",
    "Benoni Quetzalcoatlus": "Бенони-кетцалькоатль",
    "Benoni-Indian Defense": "Бенони-индийская защита",
    "Berlin Wall": "Берлинская стена",
    "Berlin Wall Defense": "Защита «Берлинская стена»",
    "Big Center Variation": "Вариант с большим центром",
    "Bird Invitation": "Приглашение Бёрда",
    "Bishop Sacrifice": "Жертва слона",
    "Breyer": "Брейер",
    "Brigg's Trap": "Ловушка Бригга",
    "Bugayev Advance Variation": "Вариант Бугаева с продвижением",
    "Brooklyn Retreat Defense": "Защита «Бруклинское отступление»",
    "Center Holding Variation": "Вариант с удержанием центра",
    "Central Break": "Центральный прорыв",
    "Central Exchange": "Центральный размен",
    "Central Storming Variation": "Вариант с штурмом центра",
    "Chigorin": "Чигорин",
    "Chigorin Plan": "План Чигорина",
    "Chinese Dragon": "Китайский дракон",
    "Christiansen's Dream": "Мечта Кристиансена",
    "Classical Fianchetto": "Классическое фианхетто",
    "Classical Main Line": "Классическая основная линия",
    "Classical Tarrasch Gambit": "Классический гамбит Тарраша",
    "Closed Berlin Defense": "Закрытая берлинская защита",
    "Closed Bernstein Variation": "Закрытый вариант Бернштейна",
    "Closed Showalter Variation": "Закрытый вариант Шоуолтера",
    "Closed Wolf Variation": "Закрытый вариант Вольфа",
    "Coles Sicilian Gambit": "Сицилианский гамбит Коула",
    "Correspondence Refutation": "Корреспондентское опровержение",
    "Czech Benoni Defense": "Чешская защита Бенони",
    "Delayed .. Nc6": "Отложенный ход .. Кc6",
    "Delayed Bishop Sacrifice": "Отложенная жертва слона",
    "Delayed Castling Line": "Линия с отложенным рокированием",
    "Delayed Exchange": "Отложенный размен",
    "Delayed Fianchetto": "Отложенное фианхетто",
    "Delayed Keres Attack": "Отложенная атака Кереса",
    "Deferred Fianchetto": "Отложенное фианхетто",
    "Double Duck Formation": "Двойная утиная формация",
    "Double Fianchetto": "Двойное фианхетто",
    "Double Fianchetto Attack": "Атака с двойным фианхетто",
    "Double Grob": "Двойное начало Гроба",
    "Double Muzio Gambit": "Двойной гамбит Музио",
    "Double Pawn Sacrifice": "Двойная жертва пешки",
    "Double Spanish": "Двойная испанская партия",
    "Double-Dutch Gambit": "Двойной голландский гамбит",
    "Dragon Transfer": "Переход в «Дракона»",
    "Dubov Italian": "Итальянское начало Дубова",
    "Dubov Tarrasch": "Тарраш Дубова",
    "Dzindzi-Indian Defense": "Дзиндзи-индийская защита",
    "Döry Indian": "Индийская защита Дёри",
    "Early Deviations": "Ранние отклонения",
    "Early b3": "Раннее b3",
    "Endgame Offer": "Предложение эндшпиля",
    "English Hybrid": "Английский гибрид",
    "English Rat": "Английская «крыса»",
    "Extended Bishop Swap": "Расширенный размен слонов",
    "Fool's Mate": "Мат дурака",
    "Franco-Hiva Gambit": "Гамбит Франко-Хива",
    "Franco-Hiva Gambit Accepted": "Гамбит Франко-Хива (принятый)",
    "Franco-Nimzowitsch Variation": "Франко-нимцовичевский вариант",
    "Franco-Sicilian Hybrid": "Франко-сицилианский гибрид",
    "French Benoni": "Французский Бенони",
    "French Connection": "Французская связь",
    "Full Symmetry": "Полная симметрия",
    "Full Symmetry Line": "Линия полной симметрии",
    "Grand Prix Pterodactyl": "Птеродактиль «Гран-при»",
    "Hedgehog": "Ёж",
    "Hungarian Dragon": "Венгерский Дракон",
    "Hyperaccelerated Dragon": "Гиперускоренный Дракон",
    "Hyperaccelerated Pterodactyl": "Гиперускоренный птеродактиль",
    "Immediate Fianchetto": "Немедленное фианхетто",
    "Inverted Hanham": "Обратный Хэмхем",
    "Italian Four Knights Variation": "Итальянский вариант четырёх коней",
    "Kieseritzky": "Кисерицкий",
    "King Pterodactyl": "Королевский птеродактиль",
    "King's Gambit Reversed": "Обращённый королевский гамбит",
    "Kingside Fianchetto": "Фианхетто на королевском фланге",
    "Kingside Move Order": "Порядок ходов на королевском фланге",
    "Kitchener Folly": "Глупость Китченера",
    "Konstantinopolsky": "Константинопольский",
    "Lasker Trap": "Ловушка Ласкера",
    "Lesser Simagin (Spassky)": "Малый вариант Симагина (Спасский)",
    "Lion's Cave": "Львиная пещера",
    "Lion's Claw": "Львиный коготь",
    "Lion's Jaw": "Львиная пасть",
    "Long Whip": "Длинный кнут",
    "Magnus Smith Trap": "Ловушка Магнуса Смита",
    "Makogonov Exchange Variation": "Разменный вариант Макогонова",
    "Maltese Falcon": "Мальтийский сокол",
    "Marshall Trap": "Ловушка Маршалла",
    "Meadow Hay Trap": "Ловушка «Сеновал»",
    "Metger Unpin": "Развязка Метгера",
    "Mokele Mbembe": "Мокеле-мбембе",
    "Modern Bc4 Variation": "Современный вариант с Сc4",
    "Modern Exchange Variation": "Современный разменный вариант",
    "Modern Main Line": "Современная основная линия",
    "Modern Bishop's Opening": "Современное начало слона",
    "Modern Pterodactyl": "Современный птеродактиль",
    "Modern Sharp Variation": "Современный острый вариант",
    "Modern Transfer": "Переход в современную защиту",
    "Moeller-Bayonet Attack": "Атака Мёллера — Байонета",
    "Monkey's Bum": "Атака «Обезьяний зад»",
    "Monticelli Trap": "Ловушка Монтителли",
    "Mortimer Trap": "Ловушка Мортимера",
    "Mujannah": "Муджанна",
    "Mazedonisch": "Македонский",
    "Myers Polish Attack": "Польская атака Майерса",
    "Neo-Arkhangelsk Variation": "Неоархангельский вариант",
    "Neo-Mongoloid Defense": "Неомонголоидская защита",
    "New England Variation": "Вариант «Новая Англия»",
    "New York System": "Нью-Йоркская система",
    "New York Variation": "Нью-йоркский вариант",
    "Nimzowitsch": "Нимцович",
    "Noah's Ark Trap": "Ловушка «Ковчег Ноя»",
    "O'Sullivan Gambit": "Гамбит О'Салливана",
    "Old Sicilian": "Старая сицилианская защита",
    "Open Berlin Defense": "Открытая берлинская защита",
    "Pachman Wing Gambit": "Крыльевой гамбит Пахмана",
    "Panno Main Line": "Основная линия Панно",
    "Panov Transfer": "Переход Панова",
    "Paulsen": "Паульсен",
    "Pawn Center Variation": "Пешечно-центровой вариант",
    "Peresypkin's Sacrifice": "Жертва Пересыпкина",
    "Pirc Connection": "Связь с защитой Перца",
    "Pirc Invitation": "Приглашение Перца",
    "Poisoned Pawn Accepted": "Вариант с отравленной пешкой (принятый)",
    "Pseudo-Austrian Attack": "Псевдоавстрийская атака",
    "Pseudo-Catalan": "Псевдокаталанское начало",
    "Pseudo-Sämisch": "Псевдоземишевский вариант",
    "Pseudo-Tarrasch": "Псевдотаррашевская защита",
    "Pseudo-Tarrasch Bishop Attack": "Псевдотаррашевская атака слона",
    "Pseudo-Tarrasch Variation": "Псевдотаррашевский вариант",
    "Queen Benoni Pterodactyl": "Ферзево-бенониевский птеродактиль",
    "Queen Fianchetto Variation": "Вариант с фианхетто ферзя",
    "Queen Pteranodon": "Ферзевый птеранодон",
    "Queen Pterodactyl": "Ферзевый птеродактиль",
    "Queen's Gambit Invitation": "Приглашение в ферзевый гамбит",
    "Queen's Knight": "Ферзевый конь",
    "Queenside Fianchetto Variation": "Вариант с фианхетто на ферзевом фланге",
    "Rare Defenses": "Редкие защиты",
    "Reversed Albin Countergambit": "Обращённый контргамбит Альбина",
    "Reversed Alekhine": "Обращённый Алехин",
    "Reversed Blumenfeld Gambit": "Обращённый гамбит Блюменфельда",
    "Reversed Brooklyn": "Обращённый Бруклин",
    "Reversed Brooklyn Defense": "Обращённая бруклинская защита",
    "Reversed Chigorin Defense": "Обращённая защита Чигорина",
    "Reversed Dragon": "Обращённый Дракон",
    "Reversed French": "Обращённая французская защита",
    "Reversed Grünfeld": "Обращённый Грюнфельд",
    "Reversed Krebs": "Обращённый Кребс",
    "Reversed Mexican Defense": "Обращённая мексиканская защита",
    "Reversed Mokele Mbembe": "Обращённый мокеле-мбембе",
    "Reversed Nimzowitsch": "Обращённый Нимцович",
    "Reversed Norwegian Defense": "Обращённая норвежская защита",
    "Reversed Rat": "Обращённая «крыса»",
    "Reversed Scandinavian": "Обращённая скандинавская защита",
    "Reversed Sicilian": "Обращённая сицилианская защита",
    "Reversed Spanish": "Обращённая испанская партия",
    "Reversed Tarrasch": "Обращённый Тарраш",
    "Reynolds' Variation": "Вариант Рейнольдса",
    "Santasiere's Folly": "Глупость Сантастьера",
    "Scotch Invitation Declined": "Шотландское приглашение (отклонённое)",
    "Sea-Cadet Mate": "Мат морского кадета",
    "Semi-Averbakh System": "Полу-система Авербаха",
    "Semi-Averbakh Variation": "Полу-вариант Авербаха",
    "Semi-Classical Variation": "Полуклассический вариант",
    "Semi-Duras Variation": "Полу-вариант Дураса",
    "Semi-Leningrad Variation": "Полуленинградский вариант",
    "Semi-Meran Variation": "Полумеранский вариант",
    "Siberian Trap": "Сибирская ловушка",
    "Sicilian Invitation": "Сицилианское приглашение",
    "Sicilian Knight Variation": "Сицилианский вариант коня",
    "Sicilian Two Knights": "Сицилианская защита с двумя конями",
    "Slav Invitation": "Славянское приглашение",
    "Small Center Defense": "Защита с малым центром",
    "Smyslov-Breyer-Zaitsev Hybrid": "Гибрид Смыслова — Брейера — Зайцева",
    "Spassky System3": "Система Спасского 3",
    "Speers": "Спирс",
    "Spanish Wing Gambit": "Испанский крыльевой гамбит",
    "Standard Development": "Стандартное развитие",
    "Sveshnikov Transfer": "Переход Свешникова",
    "Steinitz Defense Deferred": "Защита Штейница (отложенная)",
    "Steinitz Deferred": "Вариант Штейница (отложенный)",
    "Schliemann Defense Deferred": "Защита Шлиеманна (отложенная)",
    "Tarrasch Trap": "Ловушка Тарраша",
    "The Goblin": "Гоблин",
    "The Potato": "Картофель",
    "The Squirrel": "Белка",
    "The Walrus": "Морж",
    "The Whale": "Кит",
    "Traditional Colle": "Традиционная система Колле",
    "Ultra-Delayed Exchange Variation": "Ультраотложенный разменный вариант",
    "Ultra-Symmetrical Variation": "Ультрасимметричный вариант",
    "Veresov": "Вересов",
    "Vienna Hybrid": "Венский гибрид",
    "West Indian Defense": "Западно-индийская защита",
    "Wurzburger Trap": "Ловушка Вурцбурга",
    "Zaitsev Hybrid": "Гибрид Зайцева",
    "de Riviere Defense": "Защита де Ривьера",
    "de Riviere Variation": "Вариант де Ривьера",
    "de Smet Gambit": "Гамбит де Смета",
    "del Rio Attack": "Атака дель Рио",
    "del Rio Variation": "Вариант дель Рио",
    "von der Lasa Gambit": "Гамбит фон дер Лазы",
    "von Hennig Gambit": "Гамбит фон Хеннига",
    "von Popiel Gambit": "Гамбит фон Поппеля",
    "l'Hermet Variation": "Вариант Лермета",
}


# ---------------------------------------------------------------------------
# Прилагательные-головы. Согласуются в роде с типом дебюта.
# ---------------------------------------------------------------------------

DESCRIPTORS: dict[str, str] = {
    "Accel": "Ускоренный",
    "Advance": "Продвинутый",
    "Agincourt": "Азенкурский",
    "Anglo-Grünfeld": "Англогрюнфельдский",
    "Anglo-Indian": "Англо-индийский",
    "Anglo-Slav": "Анго-славянский",
    "Anti": "Анти",
    "Austrian": "Австрийский",
    "Baltic": "Балтийский",
    "Barmen": "Барменский",
    "Bayonet": "Штыковой",
    "Berlin": "Берлинский",
    "Bogo-Indian": "Бого-индийский",
    "Bishop's": "Слоновый",
    "Caro-Kann": "Каро-Каннский",
    "Catalan": "Каталанский",
    "Catalan Open": "Каталанский открытый",
    "Center": "Центральный",
    "Central": "Центральный",
    "Classical": "Классический",
    "Closed": "Закрытый",
    "Czech": "Чешский",
    "Delayed": "Отложенный",
    "Dragon": "Дракон",
    "Dutch": "Голландский",
    "East": "Восточный",
    "Eastern": "Восточный",
    "Endgame": "Эндшпильный",
    "English": "Английский",
    "Exchange": "Разменный",
    "Fianchetto": "Фианхетто",
    "Four Knights": "Четырёхконевый",
    "Four Pawns": "Четырёхпешечный",
    "French": "Французский",
    "Full": "Полный",
    "Giuoco Piano": "Джиоко Пиано",
    "Hungarian": "Венгерский",
    "Hypermodern": "Гиперсовременный",
    "Indian": "Индийский",
    "Italian": "Итальянский",
    "King's English": "Королевский английский",
    "King's Indian": "Староиндийский",
    "King's Pawn": "Королевской пешки",
    "Kingside": "Королевский фланг",
    "Latvian": "Латышский",
    "London": "Лондонский",
    "Main": "Основной",
    "Maroczy": "Мароци",
    "Modern": "Современный",
    "Neo-Grünfeld": "Неогрюнфельдский",
    "Neo-Catalan": "Неокаталанский",
    "Nimzo-Indian": "Нимцо-индийский",
    "Nimzo-Dutch": "Нимцо-голландский",
    "Normal": "Нормальный",
    "Old": "Старый",
    "Open": "Открытый",
    "Orthodox": "Ортодоксальный",
    "Philidor": "Филидоров",
    "Polish": "Польский",
    "Positional": "Позиционный",
    "Pterodactyl": "Птеродактильный",
    "Queen's English": "Ферзевый английский",
    "Queen's Indian": "Ферзево-индийский",
    "Queen's Pawn": "Ферзевой",
    "Quiet": "Тихий",
    "Reversed": "Обращённый",
    "Retreat": "С отступлением",
    "Russian": "Русский",
    "Scandinavian": "Скандинавский",
    "Scotch": "Шотландский",
    "Semi-Slav": "Полуславянский",
    "Semi-Tarrasch": "Полутаррашский",
    "Sicilian": "Сицилианский",
    "Slav": "Славянский",
    "Spanish": "Испанский",
    "Symmetrical": "Симметричный",
    "Three Knights": "Трёхконевый",
    "Two Knights": "Двухконевый",
    "Traditional": "Традиционный",
    "Vienna": "Венский",
    "Wing": "Крыльевой",
    "Yugoslav": "Югославский",
}


# ---------------------------------------------------------------------------
# Имена игроков в родительном падеже.
# ---------------------------------------------------------------------------

EONYMS: dict[str, str] = {
    "Adams": "Адамса",
    "Alekhine": "Алехина",
    "Alapin": "Алапина",
    "Albin": "Альбина",
    "Anderssen": "Андерсена",
    "Averbakh": "Авербаха",
    "Barnes": "Барнса",
    "Benko": "Бенко",
    "Berger": "Бергера",
    "Bernstein": "Бернштейна",
    "Bird": "Бёрда",
    "Blackmar": "Блэкмара",
    "Blumenfeld": "Блюменфельда",
    "Bogoljubow": "Боголюбова",
    "Boleslavsky": "Болеславского",
    "Botvinnik": "Ботвинника",
    "Breyer": "Брейера",
    "Brinckmann": "Бринкманна",
    "Bronstein": "Бронштейна",
    "Capablanca": "Капабланки",
    "Carlsbad": "Карлсбада",
    "Carls": "Карлса",
    "Chigorin": "Чигорина",
    "Danish": "Датский",
    "Diemer": "Димера",
    "Duras": "Дураса",
    "Englund": "Энглунда",
    "Evans": "Эванса",
    "Falkbeer": "Фалькбера",
    "Flohr": "Флора",
    "From": "Фрома",
    "Gligoric": "Глигорича",
    "Goldsmith": "Голдсмита",
    "Greco": "Греве",
    "Grünfeld": "Грюнфельда",
    "Gunderam": "Гундерама",
    "Gurevich": "Гуревича",
    "Hanham": "Хэмхема",
    "Harrwitz": "Харвица",
    "Henneberger": "Хеннебергера",
    "Horwitz": "Горвица",
    "Ilyin-Zhenevsky": "Ильина — Женевского",
    "Jaenisch": "Яниша",
    "Janowski": "Яновского",
    "Jobava": "Джобавы",
    "Kan": "Кана",
    "Kádas": "Кадаша",
    "Karpov": "Карпова",
    "Kasparov": "Каспарова",
    "Kazan": "Казани",
    "Kennedy": "Кеннеди",
    "Keres": "Кереса",
    "Kieseritzky": "Кисерицкого",
    "King": "короля",
    "Krause": "Краусе",
    "Larsen": "Ларсена",
    "Lasker": "Ласкера",
    "Lemberger": "Лембергера",
    "Levenfish": "Левенфиша",
    "Leonhardt": "Леонгарта",
    "MacCutcheon": "МакКатчона",
    "Maroczy": "Мароци",
    "Marshall": "Маршалла",
    "Max Lange": "Макса Ланге",
    "McCutcheon": "МакКатчона",
    "McDonnell": "Мак-Доннелла",
    "Meran": "Мерана",
    "Mieses": "Мизеса",
    "Mikenas": "Микенаса",
    "Milner-Barry": "Милнера — Барри",
    "Morphy": "Морфи",
    "Najdorf": "Найдорфа",
    "Nimzo": "Нимцовича",
    "Nimzowitsch": "Нимцовича",
    "Noa": "Ноа",
    "Nyezhmetdinov": "Нежметдинова",
    "O'Kelly": "О'Келли",
    "Opocensky": "Опоценского",
    "Owen": "Оуэна",
    "Panov": "Панова",
    "Panteldakis": "Пантелдакиса",
    "Paulsen": "Паульсена",
    "Pelikan": "Пеликана",
    "Petrosian": "Петросяна",
    "Petrov": "Петрова",
    "Philidor": "Филидора",
    "Pirc": "Перца",
    "Polerio": "Полерио",
    "Ponziani": "Понциани",
    "Rapport": "Раппорта",
    "Rauzer": "Раузера",
    "Réti": "Рети",
    "Reshevsky": "Решевского",
    "Richter": "Рихтера",
    "Robatsch": "Робача",
    "Romanishin": "Романишина",
    "Rossolimo": "Россолимо",
    "Rubinstein": "Рубинштейна",
    "Salvio": "Сальвио",
    "Sämisch": "Земиша",
    "Saragossa": "Сарагоссы",
    "Schlechter": "Шлехтера",
    "Schliemann": "Шлиеманна",
    "Scheveningen": "Схевенингена",
    "Smyslov": "Смыслова",
    "Smith": "Смита",
    "Sokolsky": "Сокольского",
    "Morra": "Морры",
    "Spassky": "Спасского",
    "Spielmann": "Шпильманна",
    "Sveshnikov": "Свешникова",
    "Stanley": "Стэнли",
    "Staunton": "Стонтона",
    "Steinitz": "Штейница",
    "Steiner": "Штейнера",
    "Stoltz": "Штольца",
    "Stonewall": "«Каменная стена»",
    "Taimanov": "Тайманова",
    "Tarrasch": "Тарраша",
    "Tartakower": "Тартаковера",
    "Torre": "Торре",
    "Trompowsky": "Тромповского",
    "Veresov": "Вересова",
    "Wade": "Уэйда",
    "Ware": "Уэра",
    "Winawer": "Винавера",
    "Yusupov": "Юсупова",
    "Zilbermints": "Зильберминца",
    "Zukertort": "Цукерторта",
}


# ---------------------------------------------------------------------------
# Транслитерация латиницы в кириллицу (для неизвестных имён игроков).
# ---------------------------------------------------------------------------

_DIGRAPHS = [
    ("tsch", "ч"), ("sch", "щ"), ("sh", "ш"), ("ch", "ч"), ("ts", "ц"), ("tz", "ц"),
    ("ck", "к"), ("ph", "ф"), ("th", "т"), ("kh", "х"), ("gh", "г"),
    ("sz", "с"), ("ss", "сс"), ("oe", "о"), ("ae", "а"), ("ue", "у"),
    ("ie", "и"), ("eu", "еу"), ("ou", "у"), ("wh", "в"),
]

_SINGLE = {
    "a": "а", "b": "б", "d": "д", "e": "е", "f": "ф", "g": "г", "h": "х",
    "i": "и", "j": "й", "k": "к", "l": "л", "m": "м", "n": "н", "o": "о",
    "p": "п", "q": "к", "r": "р", "s": "с", "t": "т", "u": "у", "v": "в",
    "w": "в", "x": "кс", "y": "и", "z": "з",
}


def _transliterate(text: str) -> str:
    """Латиница → кириллица. Апострофы, дефисы и точки сохраняются."""
    out: list[str] = []
    i = 0
    s = text.lower()
    while i < len(s):
        ch = s[i]
        if not ch.isalpha():
            out.append(ch)
            i += 1
            continue
        if ch == "c":
            nxt = s[i + 1] if i + 1 < len(s) else ""
            out.append("с" if nxt in "eiыи" else "к")
            i += 1
            continue
        for digraph, ru in _DIGRAPHS:
            if s.startswith(digraph, i):
                out.append(ru)
                i += len(digraph)
                break
        else:
            out.append(_SINGLE.get(ch, ch))
            i += 1
    return "".join(out)


def _cap(text: str) -> str:
    """Поднимает регистр первой буквы, сохраняя остальное как есть."""
    return text[:1].upper() + text[1:] if text else text


def _genitive(ru_name: str) -> str:
    """Переводит русское имя в родительный падеж («вариант <Имя>»)."""
    if not ru_name:
        return ru_name
    word = ru_name.split()[-1]
    prefix = ru_name[: len(ru_name) - len(word)]

    if word.endswith(("ский", "цкий", "ный", "кий", "ый", "ий")):
        stem = word[:-2]
        return prefix + stem + "ого"
    if word.endswith("й"):
        return prefix + word[:-1] + "я"
    if word.endswith("ь"):
        return prefix + word[:-1] + "я"
    if word.endswith("ы"):
        return prefix + word[:-1] + "а"
    if word.endswith("а"):
        stem = word[:-1]
        soft = stem[-1:] in "гкхжчшщ"
        return prefix + stem + ("и" if soft else "ы")
    if word.endswith("я"):
        return prefix + word[:-1] + "и"
    if word and word[-1] in "оеёэуюи":
        return ru_name  # несклоняемое
    return ru_name + "а"


def _agree_adjective(adj: str, kind: str) -> str:
    """Согласует прилагательное в роде с типом дебюта.

    «Классический» → «Классическая» / «Классическое», «Слоновий» →
    «Слоновья» / «Слоновье» (после «в» мягкий знак остаётся).
    """
    gender = _KIND_GENDER.get(kind, "m")
    if gender == "m" or not adj:
        return adj
    ending = "ья" if gender == "f" else "ье"
    if adj.endswith("вий"):
        return adj[:-2] + ending
    if adj.endswith(("ий", "ый", "ой")):
        return adj[:-2] + ("ая" if gender == "f" else "ое")
    return adj


def _with_kind(descriptor: str, kind: str) -> str:
    """«Классический» + «защита» → «Классическая защита».

    Фразы («С отступлением», «Королевской пешки») возвращаются как есть —
    к ним тип дебюта уже подобран вручную.
    """
    agreed = _agree_adjective(descriptor, kind)
    if " " in agreed:
        return agreed
    return f"{agreed} {kind}"


def _translate_moves(rest: str) -> str | None:
    """«with Nf3» → «с Kf3», «with .. d6» → «.. d6», «with b3 and Nf3»."""
    text = rest.strip()
    if " and " in text:
        parts = [_translate_moves(p.strip()) for p in text.split(" and ")]
        if any(p is None for p in parts):
            return None
        return " и ".join(parts)

    dots = ""
    while text.startswith("."):
        dots += "."
        text = text[1:]
    text = text.strip()
    if not text or not _MOVE_CHARS.match(text):
        return None

    moves = "".join(_PIECE_LETTERS.get(ch, ch) for ch in text)
    if dots:
        return (".. " if len(dots) >= 2 else "") + moves
    return "с " + moves


def _translate_head(head: str, kind: str) -> str | None:
    """Переводит голову («French», «Najdorf») вместе с типом дебюта."""
    text = head.strip()
    if not text:
        return None

    # 1. Прилагательное — согласуем в роде и добавляем тип дебюта.
    #    Фразы вроде «С отступлением» тип не берут.
    if text in DESCRIPTORS:
        return _with_kind(DESCRIPTORS[text], kind)

    # 2. Префикс («Anglo-Scandinavian», «Pseudo-Spanish») — хвост обязан быть
    #    прилагательным, иначе составное имя не переводим.
    for prefix, ru_prefix in _PREFIXES.items():
        if text.startswith(prefix):
            rest = text[len(prefix):].strip()
            if rest in DESCRIPTORS:
                agreed = _agree_adjective(DESCRIPTORS[rest], kind)
                joined = ru_prefix + agreed.lower()
                if " " in agreed:
                    return joined
                return f"{joined} {kind}"
            return None

    # 3. Имя игрока в словаре — родительный падеж.
    name = text[:-2] if text.endswith("'s") else text
    if name in EONYMS:
        return f"{kind} {EONYMS[name]}"

    # 4. Составное имя из нескольких слов/дефисов («Richter-Rauzer»).
    parts = re.split(r"[\s-]", name)
    if len(parts) > 1 and all(p and p[0].isupper() for p in parts):
        if any(p in DESCRIPTORS for p in parts):
            return None
        genitives = [_genitive(_cap(_transliterate(p))) for p in parts]
        return f"{kind} {' — '.join(genitives)}"

    # 5. Неизвестное односложное имя — транслитерируем.
    if len(name) > 1 and name[0].isupper() and "'" not in name:
        return f"{kind} {_genitive(_cap(_transliterate(name)))}"

    return None


def _translate_component(component: str) -> str | None:
    """Переводит один компонент имени. ``None`` — перевести не удалось."""
    text = component.strip()
    if not text:
        return ""

    # 1. Точное совпадение со словарём подвариантов.
    if text in COMPONENTS:
        return COMPONENTS[text]

    # 2. Базовая семья дебюта — первый компонент имени.
    if text in FAMILIES:
        return FAMILIES[text]

    # 3. «Steinitz Defense Deferred» → «защита Штейница (отложенная)».
    if text.endswith(" Deferred"):
        translated = _translate_component(text[: -len(" Deferred")].strip())
        return f"{translated} (отложенная)" if translated else None

    # 4. «with ...» — ход, серия ходов или фраза с типом дебюта.
    with_match = _WITH_RE.match(text)
    if with_match:
        rest = with_match.group(1).strip()
        moves = _translate_moves(rest)
        if moves is not None:
            return moves
        # «with Max Lange Defense» → «с защитой Макса Ланге».
        translated = _translate_component(rest)
        if translated:
            parts = translated.split(None, 1)
            if len(parts) == 2 and parts[0] in _KIND_INSTRUMENTAL:
                return f"с {_KIND_INSTRUMENTAL[parts[0]]} {parts[1]}"
        return None

    # 5. Правила суффиксов: «Najdorf Variation», «French Defense».
    for suffix, kind in _SUFFIX_KINDS.items():
        if text.endswith(" " + suffix):
            head = text[: -(len(suffix) + 1)].strip()
            return _translate_head(head, kind)

    # 6. «<Что-то> Accepted/Declined» → «... (принятый)».
    for word, label in _ACCEPTED_DECLINED.items():
        if text.endswith(" " + word):
            head = text[: -(len(word) + 1)].strip()
            translated = _translate_component(head)
            if translated:
                return f"{translated} ({label})"
            return None

    # 7. Одиночное прилагательное («Closed», «Open», «Classical»).
    if text in DESCRIPTORS:
        return _agree_adjective(DESCRIPTORS[text], "защита")

    # 8. Голый ход вместо «with»: «a6», «h6», «Be7».
    if len(text) >= 2 and any(c.isdigit() for c in text) and _MOVE_CHARS.match(text):
        return _translate_moves(text)

    # 9. Одиночное имя без типа дебюта: «Allgaier» → «Алльгайер».
    #    Известные имена игроков лежат в COMPONENTS — здесь только неизвестные.
    if 1 < len(text) <= 30 and text[0].isupper() and all(
        c.isalpha() or c in "'’-" for c in text
    ):
        return _cap(_transliterate(text))

    return None


def translate_opening_name(name: str | None) -> str | None:
    """Переводит полное название дебюта на русский.

    Возвращает ``None``, если хотя бы один компонент не удалось перевести —
    тогда потребитель показывает оригинальное английское имя.
    """
    if not name or not name.strip():
        return None

    tokens = _SPLIT_RE.split(name.strip())
    result: list[str] = []

    for token in tokens:
        if not token:
            continue
        # Разделители (": " / ", ") — переносим как есть.
        if _SPLIT_RE.fullmatch(token):
            result.append(token)
            continue

        translated = _translate_component(token)
        if translated is None:
            return None
        # Ходы («с e3», «.. d6») остаются со строчной, остальное — с заглавной.
        if not translated.startswith(("с ", "..")):
            translated = translated[:1].upper() + translated[1:]
        result.append(translated)

    out = "".join(result).strip()
    return out or None
