"""Fixed parsers. Protocol compliance and recoverable content are separate."""
import re
import chess

UCI = re.compile(r'(?<![a-z0-9])([a-h][1-8][a-h][1-8][qrbn]?)(?![a-z0-9])')
NEGATIVE = re.compile(r'\b(?:не|нет|или|вместо|нельзя|not|instead|or)\b', re.I)
PIECES = {'p':r'пешк\w*','n':r'(?:конь|коня|коню|коне|конем|кони)',
          'b':r'слон\w*','r':r'ладь\w*','q':r'ферз\w*','k':r'корол\w*'}


def clean(text):
    return text.strip().strip(chr(96)+'_*').strip()


def _parse_bool(text):
    text=clean(text).lower().replace('ё','е')
    labels=re.findall(r'\b(?:да|нет|yes|no)\b',text)
    if len(set(labels))!=1 or not re.match(r'^(?:ответ\s*:\s*)?(да|нет|yes|no)\b',text):
        return None
    return labels[0] in ('да','yes')


def _parse_move(fen,text,allow_embedded=True):
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
    if not allow_embedded:return None
    tokens=UCI.findall(text)
    if len(tokens)!=1 or NEGATIVE.search(text):return None
    move=chess.Move.from_uci(tokens[0])
    return move.uci() if move in board.legal_moves else None


def _parse_piece(text,square):
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


def _parse_set(text):
    """Return None for malformed text, [] for an explicit empty move list."""
    s=clean(text).lower().strip()
    if re.fullmatch(r'(?:нет|none|no|нет ходов|ходов нет|нет легальных ходов)\.?',s) or s=='[]':return []
    s=re.sub(r'^(?:all legal moves|legal moves|все легальные ходы|легальные ходы|ходы)\s*:\s*','',s)
    if s.startswith('[')!=s.endswith(']'):return None
    s=s.strip('[]')
    if not s:return None
    parts=[p.strip(' "\'') for p in re.split(r'[,\s;]+',s) if p.strip()]
    if not parts or any(not re.fullmatch(r'[a-h][1-8][a-h][1-8][qrbn]?',p) for p in parts):return None
    return parts


def parse_choice(text,alphabet='ABCD'):
    """Recover one explicit selection without treating an option list as an answer."""
    s=re.sub(r'[`*_]','',text).strip()
    letters=alphabet+alphabet.lower()
    simple=re.sub(r'^(?:ответ|answer)\s*:\s*','',s,flags=re.I).rstrip('.').strip()
    if re.fullmatch('['+letters+']',simple):return simple.upper()
    choices={x.upper() for x in re.findall(r'(?<!\w)['+letters+r'](?!\w)',s)}
    if len(choices)!=1:return None
    if re.search(r'\b(?:не|not|вместо|неверн\w*)\s+(?:вариант\s+)?['+letters+r']\b',s,re.I):return None
    direct=re.match(r'^(?:(?:ответ|answer|выбираю|выбор|вариант|option)\s*[:—-]?\s*)?(['+letters+r'])\s*[).:]',s,re.I)
    stated=re.match(r'^(?:верный|правильный|correct)\s+(?:комментарий|ответ|вариант|answer|option)\b[^:\n]{0,180}[:—–-]\s*(['+letters+r'])(?:\s*[).:]|\s*$)',s,re.I)
    match=direct or stated
    return match[1].upper() if match else None


def _parse_scalar(kind,text):
    s=re.sub(r'^(?:Ответ|Answer)\s*:\s*','',clean(text),flags=re.I).rstrip('.').strip()
    if kind=='int':
        s=s.replace('−','-')
        boxed=re.fullmatch(r'\$?\\boxed\{(-?\d+)\}\$?',s)
        if boxed:s=boxed[1]
        return int(s) if re.fullmatch(r'-?\d+',s) else None
    if kind=='choice':return parse_choice(text)
    if kind=='judgement':return parse_choice(text,'ABCDEFG')
    raise ValueError(kind)


def final_answers(text):
    """Explicit answer blocks only; quoted/fenced examples are not conclusions."""
    normalized=re.sub(r'[*_]','',text)
    masked=[];fence_char=None;fence_length=0
    for line in normalized.splitlines(keepends=True):
        marker=re.match(r'^\s*(`{3,}|~{3,})(.*)$',line.rstrip('\r\n'))
        if marker and fence_char is None:
            fence_char=marker[1][0];fence_length=len(marker[1])
            masked.append(re.sub(r'[^\r\n]',' ',line));continue
        if marker and marker[1][0]==fence_char and len(marker[1])>=fence_length and not marker[2].strip():
            fence_char=None;fence_length=0
            masked.append(re.sub(r'[^\r\n]',' ',line));continue
        masked.append(re.sub(r'[^\r\n]',' ',line) if fence_char or line.lstrip().startswith('>') else line)
    pattern=r'(?im)^[ \t]*(?:#{1,6}[ \t]*)?(?:\d+[.)][ \t]*)?(?:[✅🎯][ \t]*)?(?:итоговый ответ|окончательный ответ|final answer|ответ|answer|итог)[ \t]*(?::[ \t]*|[ \t]*\n)'
    matches=list(re.finditer(pattern,''.join(masked)))
    return [normalized[m.end():matches[i+1].start() if i+1<len(matches) else len(normalized)].strip()
            for i,m in enumerate(matches)]


def recover(text,parser):
    blocks=final_answers(text)
    if not blocks:return parser(text)
    values=[parser(block) for block in blocks]
    return values[0] if all(v is not None and v==values[0] for v in values) else None


def typed_final(value,parser):
    result=parser(value)
    if result is not None:return result
    lines=value.splitlines()
    if len(lines)<2:return None
    rest='\n'.join(lines[1:]).strip()
    header=re.match(r'^(?:#{1,6}\s*)?(?:обоснование|пояснение|explanation|reasoning)\s*:',rest,re.I)
    if not header:return None
    explanation=rest[header.end():].strip()
    if re.search(r'(?im)(?:^|[.!?\n])\s*(?:(?:или|вместо|однако|но|исправл\w*|на самом деле)\b|нет[,!.])',explanation):return None
    return parser(lines[0])


def parse_bool(text):
    return recover(text,_parse_bool)


def parse_move(fen,text):
    if final_answers(text):
        return recover(text,lambda value:typed_final(value,lambda part:_parse_move(fen,part)))
    return _parse_move(fen,text)


def parse_piece(text,square):
    value=recover(text,lambda part:_parse_piece(part,square))
    if value is not None or final_answers(text):return value
    # Repeated, agreeing target assertions are recoverable; conflicting claims are not.
    parts=[part.strip() for part in re.split(r'\n\s*\n',re.sub(r'[*_]','',text))]
    claims=[_parse_piece(part,square) for part in parts
            if re.match(r'(?:на\s+(?:клетке\s+|поле\s+)?|клетка\s+|поле\s+)?'+re.escape(square)+r'\b',part,re.I)]
    return claims[0] if claims and all(v is not None and v==claims[0] for v in claims) else None


def parse_set(text):
    return recover(text,_parse_set)


def parse_scalar(kind,text):
    def typed(value):
        if kind in ('int','judgement'):
            value=value.replace('−','-')
            return typed_final(value,lambda part:_parse_scalar(kind,part))
        return _parse_scalar(kind,value)
    # Outside an explicit answer block retain strict scalar parsing.
    return recover(text,typed) if final_answers(text) else _parse_scalar(kind,text)


def first_answer_fragment(case,text,truncated=False):
    """A complete first answer line; later explicit answers must agree."""
    text=text.replace('\r\n','\n').replace('\r','\n')
    lines=[line for line in text.splitlines(keepends=True) if line.strip()]
    if not lines or (truncated and not lines[0].endswith(('\n','\r'))):
        return ''
    def fragment(value):
        line=value.splitlines()[0] if value.splitlines() else ''
        line=re.sub(r'[*_]', '', line).strip()
        return re.sub(r'^(?:ответ|answer)\s*:\s*','',line,flags=re.I).strip()
    def parsed(value):
        kind=case['score_type']
        if kind=='bool':return _parse_bool(value)
        if kind=='piece':return _parse_piece(value,case['gold']['square'])
        if kind=='move_set':
            moves=_parse_set(value)
            return frozenset(moves) if moves is not None else None
        if kind in ('human_move','engine_move','mate_move','puzzle_move'):
            move=_parse_move(case['fen'],value,allow_embedded=False)
            return move if move is not None else (False if kind=='mate_move' and _parse_bool(value) is False else None)
        return _parse_scalar(kind,value)
    first=fragment(lines[0])
    value=parsed(first)
    if value is None:return ''
    complete_text=text
    if truncated and not text.endswith(('\n','\r')):
        complete_text=text.rsplit('\n',1)[0]
    blocks=final_answers(complete_text)
    for index,block in enumerate(blocks):
        if truncated and index==len(blocks)-1 and not block.strip():continue
        if parsed(fragment(block))!=value:return ''
    return first


def compliant(kind,text):
    s=clean(text).strip()
    if kind in ('human_move','engine_move','mate_move','puzzle_move'):
        return bool(re.fullmatch(r'[a-h][1-8][a-h][1-8][qrbn]?\.?|нет\.?',s))
    if kind=='bool':return bool(re.fullmatch(r'(?:да|нет|yes|no)\.?',s,re.I))
    if kind=='choice':
        value=re.sub(r'^(?:ответ|answer)\s*:\s*','',s,flags=re.I).rstrip('.').strip()
        return bool(re.fullmatch('[ABCDabcd]',value))
    if kind=='int':return _parse_scalar(kind,text) is not None
    if kind=='judgement':
        value=re.sub(r'^(?:ответ|answer)\s*:\s*','',s,flags=re.I).rstrip('.').strip()
        return bool(re.fullmatch('[A-Ga-g]',value))
    if kind=='piece':
        squares=set(re.findall(r'(?<![a-z0-9])[a-h][1-8](?![a-z0-9])',s.lower()))
        return len(squares)==1 and _parse_piece(s,next(iter(squares))) is not None
    if kind=='move_set':
        parsed=_parse_set(text)
        return parsed is not None and len(parsed)==len(set(parsed))
    return bool(s)
