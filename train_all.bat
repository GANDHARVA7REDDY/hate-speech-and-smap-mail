@echo off
call .venv\Scripts\activate
python -m src.train --task all --epochs 3 --batch-size 8
pause
