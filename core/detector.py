# core/detector.py
import os
from ultralytics import YOLO


class LayoutDetector:
    def __init__(self, weights_path='models/best_seg.pt'):
        """Инициализация детектора YOLOv8"""
        if os.path.exists(weights_path):
            self.model = YOLO(weights_path)
            print(f"✅ Успешно загружена модель детектора: {weights_path}")
        else:
            self.model = None
            print(f"⚠️ Файл весов детектора не найден по пути: {weights_path}")

    def detect(self, image_path, conf=0.05):
        """
        Запускает инференс и возвращает список аннотаций (полигонов)
        в формате, совместимом с фронтендом.
        """
        if self.model is None:
            return []

        annotations = []
        try:
            # Запускаем инференс YOLOv8
            results = self.model(image_path, conf=conf)
            result = results[0]

            if result.boxes is not None and len(result.boxes) > 0:
                class_names = self.model.names
                has_masks = result.masks is not None

                for i, box in enumerate(result.boxes):
                    cls_id = int(box.cls[0])
                    raw_label = class_names.get(cls_id, 'wall').lower()

                    # Маппинг классов для фронтенда
                    if 'wall' in raw_label:
                        label = 'wall'
                    elif 'wind' in raw_label:
                        label = 'window'
                    elif 'door' in raw_label:
                        label = 'door'
                    else:
                        label = 'wall'

                    # Получаем точки полигона (маска или углы bounding box)
                    if has_masks:
                        points = result.masks.xy[i].tolist()
                    else:
                        x1, y1, x2, y2 = map(float, box.xyxy[0])
                        points = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

                    # Добавляем только валидные полигоны
                    if len(points) >= 3:
                        annotations.append({
                            'label': label,
                            'points': points
                        })
        except Exception as e:
            print(f" Ошибка во время работы детектора YOLO: {e}")

        return annotations