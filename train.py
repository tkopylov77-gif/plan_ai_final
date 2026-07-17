# train.py
from ultralytics import YOLO
import torch

def main():
    # 1. Проверяем наличие видеокарты NVIDIA для аппаратного ускорения (CUDA)
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f" Обучение будет запущено на: {device}")

    # 2. Загружаем предобученную на COCO модель сегментации (yolov8n-seg.pt)
    # Суффикс -seg критически важен: он указывает, что мы учим модель рисовать полигоны, а не просто рамки
    model = YOLO('yolov8n-seg.pt')

    # 3. Запускаем обучение на наших данных
    model.train(
        data='floor_plan_training/data.yaml',  # Путь к созданному data.yaml
        epochs=100,                            # Количество эпох (оптимально 50-100 для хорошей точности)
        imgsz=640,                             # Размер картинок на входе в сеть (стандарт для YOLO)
        batch=8,                               # Количество картинок в одном пакете (уменьшите до 4 или 2 при нехватке памяти)
        device=device,                         # Устройство вычислений (0 для GPU или 'cpu')
        workers=2,                             # Потоки загрузки данных (для Windows стабильнее ставить 2 или 0)
        project='runs/segment',                # Папка для сохранения результатов
        name='floor_plan_yolo'                 # Название папки этого запуска
    )

if __name__ == '__main__':
    main()