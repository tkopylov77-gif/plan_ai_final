import os
import json
import time
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename

# Импортируем наши модули ядра
from core.detector import LayoutDetector
from core.generator import LayoutGenerator

app = Flask(__name__)

# Настройки папок
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['GENERATED_FOLDER'] = 'static/generated'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Лимит 16 МБ

# Создаем директории при старте
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['GENERATED_FOLDER'], exist_ok=True)

# -------------------------------------------------------------------------
# Инициализация моделей по абсолютным путям
# -------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Инициализируем детектор YOLOv8
YOLO_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best_seg.pt')
detector = LayoutDetector(YOLO_MODEL_PATH)

# Инициализируем генератор U-Net / Mock
GENERATOR_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'generator.pt')
generator = LayoutGenerator(GENERATOR_MODEL_PATH)


# -------------------------------------------------------------------------
# Маршруты (Эндпоинты)
# -------------------------------------------------------------------------

@app.route('/')
def index():
    """Главная страница веб-интерфейса"""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """Загрузка чертежа и его авто-разметка через класс Detector"""
    if 'file' not in request.files:
        return jsonify({'error': 'Файл чертежа не найден в запросе'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Файл не выбран'}), 400

    # Сохраняем чертеж на диск
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Запускаем наш красивый детектор в одну строчку кода!
    annotations = detector.detect(filepath, conf=0.15)

    # Приводим путь к единому веб-стандарту
    web_image_path = filepath.replace("\\", "/")

    return jsonify({
        'image_url': f'/{web_image_path}',
        'annotations': annotations
    })


@app.route('/generate', methods=['POST'])
def generate_layout():
    """Генерация новой планировки по параметрам площади и формы (U-Net / Mock)"""
    data = request.get_json() or {}
    shape_type = data.get('shape_type', 'square')
    area = float(data.get('area', 60))

    try:
        # Генерация планировки
        input_img, output_img = generator.generate(shape_type, area)

        # Уникальный таймстамп исключает кэширование изображений браузером
        ts = int(time.time())

        input_filename = f"input_{shape_type}_{int(area)}_{ts}.png"
        output_filename = f"output_{shape_type}_{int(area)}_{ts}.png"

        input_path = os.path.join(app.config['GENERATED_FOLDER'], input_filename)
        output_path = os.path.join(app.config['GENERATED_FOLDER'], output_filename)

        # Сохранение на диск
        input_img.save(input_path)
        output_img.save(output_path)

        # Форматирование путей для фронтенда
        web_input_path = input_path.replace("\\", "/")
        web_output_path = output_path.replace("\\", "/")

        return jsonify({
            'success': True,
            'input_url': f'/{web_input_path}',
            'output_url': f'/{web_output_path}'
        })

    except Exception as e:
        print(f"Ошибка генерации планировки: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/export', methods=['POST'])
def export_coco():
    """Экспорт готовой векторной разметки с холста в COCO JSON формат"""
    try:
        coco_data = request.get_json()
        export_path = os.path.join('static', 'coco_dataset.json')

        with open(export_path, 'w', encoding='utf-8') as f:
            json.dump(coco_data, f, ensure_ascii=False, indent=4)

        return send_file(export_path, as_attachment=True, download_name="coco_dataset.json")
    except Exception as e:
        print(f"Ошибка экспорта COCO dataset: {e}")
        return jsonify({'error': f"Не удалось сформировать JSON: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)