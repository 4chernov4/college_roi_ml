from train_model import train_model
from evaluate import evaluate_model
from roi_calculation import calculate_roi


def main():
    print("Запуск проекта: Рейтинг колледжей по ROI\n")

    print("1. Обучение модели...")
    train_model()

    print("\n2. Оценка качества модели...")
    evaluate_model()

    print("\n3. Расчёт ROI и формирование рейтинга...")
    calculate_roi()

    print("\nПроект успешно завершён")


if __name__ == "__main__":
    main()
