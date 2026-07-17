# debug_model.py
import os
import torch
from ultralytics import YOLO

# Настройки путей
MODEL_PATH = 'models/best_seg.pt'
# Укажите здесь имя любой вашей картинки из папки raw_data
TEST_IMAGE = 'raw_data/plan_00'

print("===  ЗАПУСК ДИАГНОСТИКИ МОДЕЛИ ===")

# 1. Проверяем видеокарту
print(f"Используется CUDA (видеокарта NVIDIA): {torch.cuda.is_available()}")

# 2. Проверяем наличие файлов
if not os.path.exists(MODEL_PATH):
    print(f" ОШИБКА: Файл модели '{MODEL_PATH}' не найден!")
    exit()
else:
    print(f" Файл модели найден: {MODEL_PATH}")

if not os.path.exists(TEST_IMAGE):
    # Если указанного файла нет, попробуем взять любой подходящий из raw_data
    if os.path.exists('raw_data'):
        files = [f for f in os.listdir('raw_data') if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if files:
            TEST_IMAGE = os.path.join('raw_data', files[0])
            print(f" Файл не найден. Используем для теста: {TEST_IMAGE}")
        else:
            print(" ОШИБКА: В папке 'raw_data' нет картинок для теста!")
            exit()
    else:
        print(" ОШИБКА: Папка 'raw_data' не существует!")
        exit()

# 3. Загружаем модель
try:
    model = YOLO(MODEL_PATH)
    print(" Модель успешно загружена в YOLO!")
except Exception as e:
    print(f" ОШИБКА при загрузке модели: {e}")
    exit()

# 4. Делаем предсказание с ультра-низким порогом уверенности (1%)
try:
    print(f"Запуск распознавания для '{TEST_IMAGE}' (conf=0.01)...")
    results = model(TEST_IMAGE, conf=0.01)
    result = results[0]

    print("\n---  КРАТКИЙ ОТЧЕТ YOLO ---")
    # Метод verbose() выведет в консоль список найденных классов, например: "1 wall, 2 windows"
    print(result.verbose() if result.verbose() else "Ничего не найдено")

    print("\n---  ДЕТАЛЬНЫЙ АНАЛИЗ ---")
    boxes = result.boxes
    masks = result.masks

    if boxes is not None and len(boxes) > 0:
        print(f"Найдено объектов (рамки bboxes): {len(boxes)}")
        for i, box in enumerate(boxes):
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            print(f"  [{i + 1}] Класс ID {cls_id} | Уверенность: {conf:.4f}")
    else:
        print(" Объекты (рамки) вообще не обнаружены!")

    if masks is not None:
        print(f"Найдено масок сегментации (masks): {len(masks)}")
        # Проверим координаты первой маски
        if len(masks.xy) > 0:
            print(f"  Первые 3 точки первой маски: {masks.xy[0][:3]}")
    else:
        print(" Маски сегментации вообще отсутствуют!")

except Exception as e:
    print(f" КРИТИЧЕСКАЯ ОШИБКА при распознавании: {e}")

print("\n=== КОНЕЦ ДИАГНОСТИКИ ===")