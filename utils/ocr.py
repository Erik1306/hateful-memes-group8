import os
import glob
import json
import easyocr
import numpy as np


def cast_pred_type(pred):
    result = []
    for tup in pred:
        coord, txt, score = tup
        coord = np.array(coord).tolist()
        score = float(score)
        result.append((coord, txt, score))
    return result


def detect(root_dir, image_dir):
    base_dir = os.path.basename(image_dir)
    out_json = os.path.join(root_dir, f'{base_dir}.jsonl')
    with open(out_json, 'w') as f:
        reader = easyocr.Reader(['en'])
        images = glob.glob(os.path.join(image_dir, '*.png'))
        images += glob.glob(os.path.join(image_dir, '**', '*.png'))
        print(f"Find {len(images)} images!")

        for i, image_path in enumerate(images):
            print(F"{i + 1}/{len(images)}")
            img_name = os.path.basename(image_path)
            pred = reader.readtext(image_path)
            result = cast_pred_type(pred)
            text = ' '.join(p[1] for p in result)
            annotation = {
                "id": int(img_name.split('.')[0]),
                "img": img_name,
                "text": text
            }
            json.dump(annotation, f)
            f.write('\n')
    return os.path.basename(out_json)
