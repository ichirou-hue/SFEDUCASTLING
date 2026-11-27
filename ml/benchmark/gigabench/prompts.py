"""Versioned, model-independent response instructions for new benchmark runs."""
import re

ANSWER_FIRST = 'answer_first_v1'
ANSWER_FIRST_INSTRUCTION = (
    'Сначала дай краткий ответ отдельной строкой. '
    'Затем, если нужно, добавь пояснение. Пояснение необязательно.'
)


def apply_answer_protocol(rows, protocol):
    if protocol not in ('legacy', ANSWER_FIRST):
        raise ValueError('Unknown answer protocol: '+str(protocol))
    if protocol == 'legacy':
        return rows
    for row in rows:
        if row['score_type'] == 'explanation':
            continue
        prompt = re.sub(r'Ответь только\b', 'Кратко ответь', row['prompt'])
        prompt = prompt.replace('Только UCI', 'Формат краткого ответа: UCI')
        prompt = prompt.replace(', без пояснений', '')
        row['prompt'] = prompt+'\n\n'+ANSWER_FIRST_INSTRUCTION
        row['answer_protocol'] = protocol
    return rows
