import argparse
import os
import sys
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from pathlib import Path
import torch
import zipfile
from model import HatefulMemesModel
from utils.ocr import detect


data_dir = Path.cwd() / 'meme_data'
train_path = data_dir / "train.jsonl"
dev_path = data_dir / "dev.jsonl"
test_path = data_dir / "test.jsonl"


class ModelNotTrained(Exception):
    pass


class NotFound(Exception):
    pass


def load_model(h_params, model_filename):
    hateful_memes_model = HatefulMemesModel(hparams=h_params)
    hateful_memes_model.load_state_dict(torch.load(model_filename))
    return hateful_memes_model


def predict(root, img_zip=None, img_dir=None) -> dict:
    root = Path.cwd() / root
    if not os.path.exists(root):
        raise NotFound(f'Directory with path {root} not found!')

    if img_dir:
        img_dir = root / img_dir
        if not os.path.exists(img_dir):
            raise NotFound(f'Directory with path {img_dir} not found!')
    elif img_zip:
        img_dir = root / img_zip.replace('.zip', '')
        img_zip = root / img_zip
        if not os.path.exists(img_zip):
            raise NotFound(f'Zip file with path {img_zip} not found!')
        if not os.path.exists(os.path.join(root, img_dir)):
            with zipfile.ZipFile(img_zip, 'r') as zip_ref:
                zip_ref.extractall(root)
    predict_json = detect(root, img_dir)

    model_path = 'saved_models/pl_model.txt'
    if not os.path.exists(model_path):
        raise ModelNotTrained(f'Trained model with {model_path} path not found!')

    h_params = {
        "train_path": train_path,
        "dev_path": dev_path,
        "img_dir": img_dir,
        "embedding_dim": 150,
        "language_feature_dim": 300,
        "vision_feature_dim": 300,
        "fusion_output_size": 256,
        "output_path": "model-outputs",
        "dev_limit": None,
        "lr": 0.00005,
        "max_epochs": 5,
        "n_gpu": 1,
        "batch_size": 4,
        "accumulate_grad_batches": 16,
        "early_stop_patience": 3,
    }

    model = load_model(h_params, model_path)
    predict_json = data_dir / predict_json
    submission = model.make_submission_frame(
        predict_json
    )
    probabilities = submission.proba
    data = {
        "probabilities": probabilities,
        "predict_json": predict_json,
    }
    return data


def get_formatted_data(probabilities, predict_json, threshold):
    img_it_to_name = {}
    with open(predict_json) as f:
        for line in f:
            content = json.loads(line)
            img_it_to_name[int(content['id'])] = content['img']
    output = {}
    for id, proba in probabilities.items():
        output[img_it_to_name[id]] = 'hateful' if proba >= threshold else 'non hateful'

    return output


def save_predictions_as_json(output, filename):
    with open(filename, 'w') as f:
        json.dump(output, f, ensure_ascii=True, indent=4, sort_keys=True)


def argv_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', metavar='Path to the root directory which include images zip or directory',
                        type=str, required=True)
    parser.add_argument('--img_zip', metavar='Name of zip file containing images from root directory', type=str)
    parser.add_argument('--img_dir', metavar='Name of directory containing images from root directory', type=str)
    parser.add_argument('--threshold', metavar='Threshold for classification', type=float, required=True)
    parser.add_argument('--output_filename', metavar='Name of output json file.', type=str, required=True)
    args = parser.parse_args()

    return {
        'root': args.root,
        'img_zip': args.img_zip,
        'img_dir': args.img_dir,
        'threshold': args.threshold,
        'output_filename': args.output_filename
    }


if __name__ == '__main__':
    arguments = argv_parser()
    data = predict(arguments['root'], arguments['img_zip'], arguments['img_dir'])
    output = get_formatted_data(**data, threshold=arguments.get('threshold'),)
    save_predictions_as_json(output, filename=arguments.get('output_filename'))
