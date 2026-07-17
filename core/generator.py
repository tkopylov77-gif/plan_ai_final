# core/generator.py
import os
import random
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageOps
from core.generator_model import LayoutGeneratorUNet


class LayoutGenerator:
    def __init__(self, weights_path='models/generator.pt'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = LayoutGeneratorUNet(in_channels=3, out_channels=3).to(self.device)

        if os.path.exists(weights_path):
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
            print(f" Успешно загружены веса генератора: {weights_path}")
        else:
            print("⚠ Веса генератора не найдены. Модуль запущен в режиме генерации псевдо-случайных шаблонов!")
            self.model = None

        # Стандартные трансформации для Pix2Pix моделей
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def draw_input_mask(self, width, height, shape_type, area):
        """
        Вспомогательный метод создания входной маски-ограничения по ТЗ (форма + площадь)
        """
        img = Image.new('RGB', (512, 512), color='white')
        draw = ImageDraw.Draw(img)

        # Переводим площадь (кв. м) в пиксели (примерный масштаб)
        scale_factor = 3.5  # 1 кв.м ≈ 3.5 пикселя линейного размера
        side = int(area * scale_factor)
        side = min(max(side, 120), 400)  # Ограничиваем размеры рамки

        # Вычисляем центрированные координаты
        start_x = (512 - side) // 2
        start_y = (512 - side) // 2
        end_x = start_x + side
        end_y = start_y + side

        if shape_type == 'square':
            draw.rectangle([start_x, start_y, end_x, end_y], outline='black', width=5)
        elif shape_type == 'l_shape':
            # Рисуем L-образную рамку
            offset = side // 2
            points = [
                (start_x, start_y),
                (end_x - offset, start_y),
                (end_x - offset, end_y - offset),
                (end_x, end_y - offset),
                (end_x, end_y),
                (start_x, end_y)
            ]
            draw.polygon(points, outline='black', fill='white')
            # Сверху обводим жирным контуром
            for i in range(len(points)):
                draw.line([points[i], points[(i + 1) % len(points)]], fill='black', width=5)
        else:
            # По умолчанию рисуем прямоугольный контур
            draw.rectangle([start_x + 30, start_y, end_x - 30, end_y], outline='black', width=5)

        return img

    def generate(self, shape_type='square', area=60):
        """Генерирует новую схему планировки по заданным ограничениям"""
        # 1. Формируем входное изображение-контур
        input_image = self.draw_input_mask(512, 512, shape_type, area)

        if self.model is not None:
            # 2. Если модель обучена, пропускаем картинку через U-Net
            input_tensor = self.transform(input_image).unsqueeze(0).to(self.device)
            self.model.eval()
            with torch.no_grad():
                output_tensor = self.model(input_tensor)

            # Денормализация тензора обратно в PIL Image
            output_tensor = output_tensor.squeeze(0).cpu()
            output_tensor = (output_tensor + 1) / 2.0  # Из [-1, 1] в [0, 1]
            output_image = transforms.ToPILImage()(output_tensor)
            # Приводим к исходному разрешению
            output_image = output_image.resize((512, 512))
        else:
            # 3. Динамический режим-заглушка (Mock) с процедурной генерацией комнат
            output_image = input_image.copy()
            draw = ImageDraw.Draw(output_image)

            # Определяем фактические габариты нарисованного контура
            # Находим ограничивающий прямоугольник (bbox) черного цвета (контура стен)
            gray = ImageOps.grayscale(input_image)
            # Инвертируем, чтобы стены стали белыми (так getbbox определит их границы)
            inverted = ImageOps.invert(gray)
            bbox = inverted.getbbox()

            if bbox:
                x1, y1, x2, y2 = bbox
                # Добавляем небольшой отступ внутрь контура стен
                pad = 6
                inner_x1, inner_y1 = x1 + pad, y1 + pad
                inner_x2, inner_y2 = x2 - pad, y2 - pad
                w, h = inner_x2 - inner_x1, inner_y2 - inner_y1

                # Генерируем случайные внутренние стены (межкомнатные)
                # Случайное деление по вертикали и горизонтали
                split_x = int(inner_x1 + w * random.uniform(0.35, 0.65))
                split_y = int(inner_y1 + h * random.uniform(0.35, 0.65))

                # Рисуем внутренние перегородки (красным цветом 'red')
                if shape_type == 'square' or shape_type == 'rectangle':
                    # Делим коробку на 3-4 случайные зоны
                    draw.line([(split_x, inner_y1), (split_x, inner_y2)], fill='red', width=3)
                    draw.line([(inner_x1, split_y), (split_x, split_y)], fill='red', width=3)
                    if random.choice([True, False]):
                        draw.line([(split_x, split_y + 40), (inner_x2, split_y + 40)], fill='red', width=3)

                elif shape_type == 'l_shape':
                    # Для L-образной формы учитываем её вырез (обычно правый верхний угол)
                    mid_x = inner_x1 + int(w * 0.5)
                    mid_y = inner_y1 + int(h * 0.5)
                    # Проводим стену по границе изгиба и делим оставшуюся часть
                    draw.line([(inner_x1, mid_y), (mid_x, mid_y)], fill='red', width=3)
                    draw.line([(mid_x, mid_y), (mid_x, inner_y2)], fill='red', width=3)

                    # Случайная дополнительная перегородка в левом крыле
                    sub_split_y = int(inner_y1 + (mid_y - inner_y1) * random.uniform(0.4, 0.6))
                    draw.line([(inner_x1, sub_split_y), (mid_x, sub_split_y)], fill='red', width=3)

                # Имитируем окна (голубые линии 'cyan' на внешних стенах)
                # Выбираем случайные места на внешней границе под окна
                window_len = 45
                if w > 150:
                    # Окно снизу
                    win_x1 = int(inner_x1 + w * random.uniform(0.2, 0.5))
                    draw.line([(win_x1, y2 - 2), (win_x1 + window_len, y2 - 2)], fill='cyan', width=5)
                    # Окно сверху
                    win_x2 = int(inner_x1 + w * random.uniform(0.5, 0.8))
                    # Для L-shape проверяем, чтобы окно не попало на вырез
                    if shape_type != 'l_shape' or win_x2 < (inner_x1 + w * 0.5):
                        draw.line([(win_x2, y1 + 2), (win_x2 + window_len, y1 + 2)], fill='cyan', width=5)

                # Имитируем межкомнатные двери (зеленые разрывы 'green')
                # Рисуем двери прямо поверх наших новых сгенерированных внутренних стен
                door_size = 24
                if shape_type in ['square', 'rectangle']:
                    # Дверь в вертикальной стене
                    door_y = int(inner_y1 + h * random.uniform(0.1, 0.4))
                    draw.line([(split_x, door_y), (split_x, door_y + door_size)], fill='green', width=4)
                    # Дверь в горизонтальной стене
                    door_x = int(inner_x1 + (split_x - inner_x1) * 0.5)
                    draw.line([(door_x, split_y), (door_x + door_size, split_y)], fill='green', width=4)
                else:
                    # Дверь для L-планировки
                    draw.line([(inner_x1 + 30, mid_y), (inner_x1 + 30 + door_size, mid_y)], fill='green', width=4)

            # Добавляем динамическую подпись с текущей площадью и типом формы
            draw.text((15, 15), f"Format: {shape_type.upper()}", fill="gray")
            draw.text((15, 30), f"Calculated Area: {area} sq.m", fill="gray")
            draw.text((15, 45), "Fallback Generator Mode", fill="red")

        return input_image, output_image