"""Fixed parsers. Protocol compliance and recoverable content are separate."""
import re
import chess

UCI = re.compile(r'(?<![a-z0-9])([a-h][1-8][a-h][1-8][qrbn]?)(?![a-z0-9])')
NEGATIVE = re.compile(r'\b(?:не|нет|или|вместо|нельзя|not|instead|or)\b', re.I)
PIECES = {'p':r'пешк\w*','n':r'(?:конь|коня|коню|коне|конем|кони)',
          'b':r'слон\w*','r':r'ладь\w*','q':r'ферз\w*','k':r'корол\w*'}


def clean(text):
    return text.strip().strip(chr(96)+'_*').strip()


def parse_bool(text):
    text=clean(text).lower().replace('ё','е')
    labels=re.findall(r'\b(?:да|нет|yes|no)\b',text)
    if len(set(labels))!=1 or not re.match(r'^(?:ответ\s*:\s*)?(да|нет|yes|no)\b',text):
        return None
    return labels[0] in ('да','yes')


def parse_move(fen,text):
    board=chess.Board(fen)
    s=clean(text).rstrip('.!?,;').strip()
    s=re.sub(r'^(?:ход|move|ответ)\s*:\s*','',s,flags=re.I).replace('–','-').replace('—','-')
    numbered=re.fullmatch(r'\d+\.{0,3}\s*([a-h][1-8][a-h][1-8][qrbn]?)',s)
    if numbered:s=numbered.group(1)
    if re.fullmatch(r'[a-h][1-8]\s*-?\s*[a-h][1-8][qrbnQRBN]?',s):
        try:
            move=chess.Move.from_uci(re.sub(r'[\s-]','',s).lower())
            return move.uci() if move in board.legal_moves else None
        except ValueError:return None
    san=s.replace('0-0','O-O')
    if san.startswith('Кр'):san='K'+san[2:]
    elif san[:1] in {'К','С','Л','Ф'}:san={'К':'N','С':'B','Л':'R','Ф':'Q'}[san[0]]+san[1:]
    try:return board.parse_san(san).uci()
    except ValueError:pass
    tokens=UCI.findall(text)
    if len(tokens)!=1 or NEGATIVE.search(text):return None
    move=chess.Move.from_uci(tokens[0])
    return move.uci() if move in board.legal_moves else None


def parse_piece(text,square):
    text=text.lower().replace('ё','е').strip()
    if '<|' in text or not text:return None
    squares=set(re.findall(r'(?<![a-z0-9])[a-h][1-8](?![a-z0-9])',text))
    if squares!={square}:return None
    types=[k for k,p in PIECES.items() if re.search(r'\b'+p+r'\b',text)]
    white=bool(re.search(r'\bбел\w*\b',text));black=bool(re.search(r'\bчерн\w*\b',text))
    empty=bool(re.search(r'\b(?:пуст\w*|свобод\w*)\b|\bнет\s+(?:никаких\s+)?фигур\w*|\bфигур\w*\s+нет|\bничего\s+нет',text))
    if re.search(r'\bне\b',text):return None
    if empty:return '.' if not types and not white and not black else None
    if len(types)!=1 or white==black or re.search(r'\bнет\b',text):return None
    return types[0].upper() if white else types[0]


def parse_set(text):
    """Return None for malformed text, [] for an explicit empty move list."""
    s=clean(text).lower().strip()
    if re.fullmatch(r'(?:нет|none|no|нет ходов|ходов нет|нет легальных ходов)\.?',s) or s=='[]':return []
    s=re.sub(r'^(?:legal moves|легальные ходы|ходы)\s*:\s*','',s)
    s=s.strip('[]')
    if not s:return None
    parts=[p.strip(' "\'') for p in re.split(r'[,\s;]+',s) if p.strip()]
    if not parts or any(not re.fullmatch(r'[a-h][1-8][a-h][1-8][qrbn]?',p) for p in parts):return None
    return parts


def parse_scalar(kind,text):
    s=re.sub(r'^(?:Ответ|Answer)\s*:\s*','',clean(text),flags=re.I).rstrip('.').strip()
    if kind=='int':return int(s) if re.fullmatch(r'-?\d+',s) else None
    if kind=='choice':return s.upper() if re.fullmatch('[ABCDabcd]',s) else None
    if kind=='judgement':return s.upper() if re.fullmatch('[A-Ga-g]',s) else None
    raise ValueError(kind)


def compliant(kind,text):
    s=clean(text).strip()
    if kind in ('human_move','engine_move','mate_move','puzzle_move'):
        return bool(re.fullmatch(r'[a-h][1-8][a-h][1-8][qrbn]?\.?|нет\.?',s))
    if kind=='bool':return bool(re.fullmatch(r'(?:да|нет|yes|no)\.?',s,re.I))
    if kind in ('int','choice','judgement'):return parse_scalar(kind,text) is not None
    if kind=='move_set':
        parsed=parse_set(text)
        return parsed is not None and len(parsed)==len(set(parsed))
    return bool(s)
