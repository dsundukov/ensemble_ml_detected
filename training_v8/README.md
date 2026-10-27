# Пакет обучения ансамбля детекции промпт-атак, версия v8

Пакет позволяет повторить обучение ансамбля с нуля и сверить результат с эталонным запуском.

## Состав
| Папка / файл | Содержимое |
|---|---|
| code/ | код обучения: train.py (точка входа), config.py (гиперпараметры), модули данных, моделей, стекинга и оценки |
| data/ | все данные, на которых обучалась модель, и проверочные выборки |
| expected/ | эталон: ensemble_config.json (пороги), eval_predictions.csv (ответы на контрольной выборке), train_log_v8.txt |
| requirements.txt | зависимости (минимальные версии) |
| requirements-lock.txt | точные версии библиотек эталонного запуска |
| reproduce.sh | повтор обучения одной командой (macOS / Linux) |
| compare_runs.py | сравнение повторного запуска с эталоном |

## Данные
| Файл | Строк | Атаки | Безобидные | Роль |
|---|---|---|---|---|
| public.csv | 37114 | 36077 | 1037 | обучение |
| public_ru.csv | 19647 | 18611 | 1036 | обучение |
| ru_promptsc_eval.csv | 216 | 62 | 154 | контрольная выборка: только оценка |
| ru_promptsc_threshold.csv | 162 | 47 | 115 | выборка для подбора порога |
| ru_promptsc_train.csv | 162 | 47 | 115 | не используется (есть версия _aug) |
| ru_promptsc_train_aug.csv | 1457 | 423 | 1034 | обучение |
| ru_synthetic_domain.csv | 265 | 76 | 189 | не используется (есть версия _aug) |
| ru_synthetic_domain_aug.csv | 2385 | 684 | 1701 | обучение |

Обучение берёт все CSV из data/, кроме ru_promptsc_eval.csv и ru_promptsc_threshold.csv; при наличии X.csv и X_aug.csv берётся X_aug.csv.

## Требования
- CPU (GPU не нужен), 16 ГБ RAM, около 2 ГБ на диске; доступ к huggingface.co при первом запуске.
- Python 3.10+.
- Базовые модели скачиваются с Hugging Face:
- `cointegrated/rubert-tiny2` — ревизия `e8ed3b0c8bbf4fb6984c3de043bf7d2f4e5969ae`
- `microsoft/Multilingual-MiniLM-L12-H384` — ревизия `6e8c1ec6b4ec4e3fc6eb7d2cd834fcd582b61daf`

## Как повторить
```bash
bash reproduce.sh
```
Вручную:
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-lock.txt
cd code
python train.py --data-dir ../data \
    --threshold-data ../data/ru_promptsc_threshold.csv \
    --eval-data ../data/ru_promptsc_eval.csv \
    --output ../artifacts_repro 2>&1 | tee ../train_log_repro.txt
cd ..
python compare_runs.py --expected expected --actual artifacts_repro
```
Время на Apple M4 Max: около 10–12 минут.

## Эталонный результат
- Пороги: threshold = 0.1134, gray_low = 0.0402, seed = 42.
- Контрольная выборка (216 строк, 62 атак), None = атака: precision 0.496 [0.41; 0.58], recall 0.952 [0.87; 0.98].
- Контрольная выборка (216 строк, 62 атак), None = нет атаки: precision 0.568 [0.46; 0.67], recall 0.806 [0.69; 0.89].

## Воспроизводимость
Эталон получен на macOS-15.3.1-arm64-arm-64bit-Mach-O, Python 3.13.9.
С теми же версиями библиотек (requirements-lock.txt) и seed = 42 на той же платформе результат совпадает полностью
(проверено повторным запуском). На другом процессоре или других версиях PyTorch возможны небольшие отличия
в весах трансформеров и, как следствие, в ответах на единичных строках; compare_runs.py показывает долю совпадений.

Точные версии эталонного запуска:
```
torch==2.12.0
transformers==5.2.0
accelerate==1.14.0
datasets==4.8.5
scikit-learn==1.7.2
numpy==2.3.5
pandas==2.3.3
safetensors==0.8.0
sentencepiece==0.2.2
protobuf==5.29.3
tokenizers==0.22.2
huggingface-hub==1.30.0
```
