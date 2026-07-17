import json
import os
import shutil
import random


def convert_multiple_files(src_dir, output_dir, train_ratio=0.8):
    # Создаем целевую структуру папок для YOLOv8-seg
    for folder in ['train/images', 'train/labels', 'val/images', 'val/labels']:
        os.makedirs(os.path.join(output_dir, folder), exist_ok=True)

    # Список поддерживаемых форматов изображений
    img_extensions = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')

    # 1: Сканируем папку raw_data и ищем пары "Картинка + JSON"
    pairs = []
    all_files = os.listdir(src_dir)
    json_files = [f for f in all_files if f.lower().endswith('.json')]

    for json_file in json_files:
        base_name = os.path.splitext(json_file)[0]
        # Ищем подходящую картинку с тем же именем
        img_file = None
        for ext in img_extensions:
            test_img_name = base_name + ext
            if test_img_name in all_files:
                img_file = test_img_name
                break

        if img_file:
            pairs.append({
                'json': json_file,
                'image': img_file,
                'base_name': base_name
            })
        else:
            print(f"⚠️ Предупреждение: Для разметки '{json_file}' не найдена картинка с именем '{base_name}'!")

    if not pairs:
        print(" Ошибка: Не найдено ни одной готовой пары Картинка + JSON в папке 'raw_data'!")
        return

    print(f" Найдено готовых пар для конвертации: {len(pairs)}")

    #2: Перемешиваем пары для случайного деления на train/val
    random.seed(42)
    random.shuffle(pairs)

    # Рассчитываем пропорции
    split_idx = max(1, int(len(pairs) * train_ratio))
    # Если пара всего одна, она пойдет и в train, и в val, чтобы YOLO не выдавала ошибку пустого набора
    if len(pairs) == 1:
        train_pairs = pairs
        val_pairs = pairs
        print(" Всего 1 пара файлов. Она скопирована и в train, и в val.")
    else:
        train_pairs = pairs[:split_idx]
        val_pairs = pairs[split_idx:]
        print(f" Распределение: {len(train_pairs)} пар в train, {len(val_pairs)} пар в val.")

    # Маппинг классов (в вебе: 1, 2, 3 -> в YOLO должны быть от 0: 0, 1, 2)
    cat_map = {1: 0, 2: 1, 3: 2}

    def process_subset(subset_pairs, subset_name):
        copied_count = 0
        for pair in subset_pairs:
            # Пути к исходным файлам
            src_img_path = os.path.join(src_dir, pair['image'])
            src_json_path = os.path.join(src_dir, pair['json'])

            # Пути к целевым файлам
            dest_img_path = os.path.join(output_dir, subset_name, 'images', pair['image'])
            dest_lbl_path = os.path.join(output_dir, subset_name, 'labels', f"{pair['base_name']}.txt")

            # 1. Копируем картинку
            shutil.copy(src_img_path, dest_img_path)

            # 2. Читаем JSON разметку и конвертируем в TXT (YOLOv8-seg)
            with open(src_json_path, 'r', encoding='utf-8') as jf:
                coco_data = json.load(jf)

            # Получаем ширину и высоту изображения
            img_info = coco_data.get('images', [{}])[0]
            width = img_info.get('width', 800)  # Фолбэк, если ширины нет
            height = img_info.get('height', 600)

            annotations = coco_data.get('annotations', [])

            with open(dest_lbl_path, 'w', encoding='utf-8') as lf:
                for ann in annotations:
                    category_id = ann.get('category_id', 1)
                    yolo_class = cat_map.get(category_id, 0)

                    seg = ann.get('segmentation', [])
                    if not seg:
                        continue

                    points = seg[0] if isinstance(seg[0], list) else seg

                    # Нормализуем координаты (делением на W и H)
                    normalized_points = []
                    for i in range(0, len(points), 2):
                        x_norm = points[i] / width
                        y_norm = points[i + 1] / height
                        normalized_points.append(f"{x_norm:.6f}")
                        normalized_points.append(f"{y_norm:.6f}")

                    line = f"{yolo_class} " + " ".join(normalized_points) + "\n"
                    lf.write(line)

            copied_count += 1
        print(f"   -> Успешно перенесено в {subset_name}: {copied_count} изображений и разметки.")

    print("\n--- Старт конвертации ---")
    process_subset(train_pairs, 'train')
    process_subset(val_pairs, 'val')
    print(f"\n Конвертация успешно завершена! Данные сохранены в папку: {output_dir}")


if __name__ == '__main__':
    convert_multiple_files(
        src_dir='raw_data',
        output_dir='floor_plan_training',
        train_ratio=0.8
    )