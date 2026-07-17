import json
import datetime


def export_to_coco(images_info, annotations_list, output_path='dataset.json'):
    """
    Формирует структуру COCO Dataset.
    images_info: список словарей [{"id": 1, "width": 800, "height": 600, "file_name": "plan.jpg"}]
    annotations_list: список словарей с полигонами и bbox.
    """
    coco_format = {
        "info": {
            "description": "ООО БИЛД ИТ - Floor Plan Dataset",
            "date_created": str(datetime.datetime.now())
        },
        "images": images_info,
        "annotations": annotations_list,
        "categories": [
            {"id": 1, "name": "wall", "supercategory": "structure"},
            {"id": 2, "name": "window", "supercategory": "structure"},
            {"id": 3, "name": "door", "supercategory": "structure"}
        ]
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(coco_format, f, indent=4, ensure_ascii=False)

    return output_path