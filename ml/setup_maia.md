# Установка Maia Chess

## Вариант A: Maia через Lc0 (рекомендуется)

Maia — это набор весов для движка Leela Chess Zero (lc0).

### 1. Скачать Lc0
- Windows: https://github.com/LeelaChessZero/lc0/releases
- Linux: `sudo apt install lc0` или собрать из исходников

### 2. Скачать веса Maia
- https://github.com/CSSLab/maia-chess/tree/master/maia_weights
- Файлы: maia-1100.pb.gz, maia-1200.pb.gz, ..., maia-1900.pb.gz
- Число = примерный рейтинг ELO игрока, которого имитирует Maia

### 3. Запуск
```bash
./lc0 --weights=maia-1500.pb.gz
```
Lc0 запустится как UCI-движок, совместимый с python-chess.

## Вариант B: Maia 2 (новая версия)
- https://github.com/CSSLab/maia2
- Требует Python + PyTorch
- Более точная модель, но сложнее в установке
