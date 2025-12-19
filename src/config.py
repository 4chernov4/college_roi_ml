RAW_DATA_PATH = "data/raw/Scorecard.csv"
PROCESSED_DATA_PATH = "data/processed/cleaned_data.csv"
MODEL_PATH = "models/income_model.pkl"
OUTPUT_RANKING_PATH = "outputs/college_ranking.csv"

FEATURE_COLUMNS = [
    "COSTT4_A",     # Полная стоимость обучения
    "ADM_RATE",     # Конкурс при поступлении
    "SAT_AVG",      # Средний балл SAT
    "C150_4",       # Доля выпускников за 4 года
    "DEBT_MDN"      # Медианный долг выпускников
]

TARGET_COLUMN = "DEP_INC_AVG" # Средний доход выпускников колледжа после обучения

META_COLUMNS = [
    "INSTNM",       # Название колледжа
    "CITY",         # Город
    "STABBR",       # Штат
    "CONTROL"       # Тип колледжа
]

RANDOM_STATE = 42
TEST_SIZE = 0.2

RF_PARAMS = {
    "n_estimators": 200, # Количество деревьев в лесу
    "max_depth": None, # Максимальная глубина дерева
    "random_state": RANDOM_STATE, # Фиксирует случайность
    "n_jobs": -1 # Количество потоков
}
