House_price: ML + DL пайплайн

Установка
    Python 3.11+

    Запусти:
    python -m pip install -r requirements.txt

Данные нужно скачать  `train.csv` и `test.csv` с https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques
и положить в папку `data/`.

Запуск
В `config.py` выбери режим `mode`:
- `validation` - кросс-валидация на train, результаты загрузятся в `results/ml_resutsN_*.xlsx`
- `inference` - обучение на всём train и ответы на test, результаты загрузятся в `results/inferenceN_*/`

Параметры моделей меняются в `config.py`.

Потом запускай python main.py


Что в папке inference
- `submission_<модель>.csv` - файл для отправки на Kaggle
- `all_predictions.csv` - ответы всех моделей
- `inference_results.xlsx` - параметры и train RMSE, колонка `kaggle_score` для ручного заполнения

Отдельно выгружается файл со всеми значениями цен на валидации для оценки ансамблей 
- `ensemble_result_` 
Оценки ансамбля идет в ноутбуке с EDA