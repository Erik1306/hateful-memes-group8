from flask import Flask, render_template, request, send_from_directory
from pathlib import Path
import zipfile
import os
import sys
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from model import HatefulMemesModel
from utils.predict import predict, get_formatted_data
from utils.test import test, get_formatted_test_data, get_metrics
import uuid
app = Flask(__name__)


PREDICTIONS = {}
TESTS = {}

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/paper')
def paper():
    return render_template('paper.html')


@app.route('/train_result', methods=['GET', 'POST'])
def train_result():
    if request.method == 'GET':
        base_dir = request.args.get('base_dir')
        images = request.args.get('images')
        train_path = request.args.get('train')
        test_path = request.args.get('test')
        dev_path = request.args.get('dev')
        return render_template('test_loading.html', base_dir=base_dir, images=images, train_path=train_path,
                               test_path=test_path, dev_path=dev_path)

    if request.method == 'POST':
        arguments = json.loads(request.data.decode("UTF-8"))
        base_dir, images, train_path, test_path, dev_path = arguments.get('base_dir'), arguments.get('images'), \
                                    arguments.get('train_path'), arguments.get('test_path'), arguments.get('dev_path')

        data_dir = Path.cwd() / base_dir

        img_tar_path = data_dir / images
        img_path = data_dir / 'img'
        train_path = data_dir / train_path
        dev_path = data_dir / dev_path
        test_path = data_dir / test_path

        # Extract images if there is need
        if not img_path.exists():
            with zipfile.ZipFile(img_tar_path, 'r') as zip_ref:
                zip_ref.extractall(img_path)

        hparams = {
            # Required hparams
            "train_path": train_path,
            "dev_path": dev_path,
            "img_dir": data_dir,

            # Optional hparams
            "embedding_dim": 150,
            "language_feature_dim": 300,
            "vision_feature_dim": 300,
            "fusion_output_size": 256,
            "output_path": "saved_models-outputs",
            "dev_limit": None,
            "lr": 0.00005,
            "max_epochs": 10,
            "n_gpu": 1,
            "batch_size": 4,

            # allows us to "simulate" having larger batches
            "accumulate_grad_batches": 16,
            "early_stop_patience": 3,
        }

        hateful_memes_model = HatefulMemesModel(hparams=hparams)
        hateful_memes_model.fit()
        return {'done': True}


@app.route('/train')
def train_output():
    return render_template('train_result.html')


@app.route('/test_result', methods=['GET', 'POST'])
def test_result():
    if request.method == 'GET':
        dir_path = request.args.get('dir_path')
        dir_name = request.args.get('dir_name')
        jsonl = request.args.get('jsonl')
        return render_template('test_loading.html', dir_name=dir_name, dir_path=dir_path, jsonl=jsonl)

    if request.method == 'POST':
        arguments = json.loads(request.data.decode("UTF-8"))
        dir_name, dir_path, jsonl = arguments.get('dir_name'), arguments.get('dir_path'), arguments.get('jsonl')
        dir_name = dir_name.replace('.zip', '')

        data = test(root=dir_path, img_zip=dir_name, annotations=jsonl)
        output = get_formatted_data(**data, threshold=request.args.get('threshold', 0.5), )
        img_dir = os.path.join(dir_path, dir_name)
        root = os.path.dirname(__file__)
        command = f"cp -r {os.path.join(os.path.dirname(root), img_dir)} {os.path.join(root, 'static')}"
        os.popen("sudo -S %s" % command, 'w').write('Root2018!\n')
        y_true, y_pred, pretty_y = get_formatted_test_data(pred=output, true_json='meme_data/dev.jsonl')
        metrics = get_metrics(y_pred, y_true)
        if not os.path.exists(os.path.join(root, 'static', dir_name)):
            os.listdir(os.path.join(root, 'static', dir_name))
        uid = str(uuid.uuid4())
        TESTS[uid] = dict(dir_name=dir_name, image_names=[], predictions=pretty_y, metrics=metrics)
        return {'done': True, 'uid': uid}


@app.route('/test')
def test_output():
    uid = request.args.get('uid')
    data = TESTS.pop(uid)
    dir_name, image_names, predictions, metrics = data.get('dir_name'), data.get('image_names'), data.get('predictions'), data.get('metrics')
    return render_template('test_result.html', dir_name=dir_name, image_names=[], predictions=predictions, metrics=metrics)


@app.route('/predict_result', methods=['GET', 'POST'])
def predict_result():
    if request.method == 'GET':
        dir_path = request.args.get('dir_path')
        dir_name = request.args.get('dir_name')
        return render_template('prediction_loading.html', dir_name=dir_name, dir_path=dir_path)

    if request.method == 'POST':
        arguments = json.loads(request.data.decode("UTF-8"))
        dir_name, dir_path = arguments.get('dir_name'), arguments.get('dir_path')
        dir_name = dir_name.replace('.zip', '')
        data = predict(root=dir_path, img_zip=dir_name)
        output = get_formatted_data(**data, threshold=request.args.get('threshold', 0.5),)
        img_dir = os.path.join(dir_path, dir_name)
        root = os.path.dirname(__file__)
        # command = f"cp -r {os.path.join(os.path.dirname(root), img_dir)} {os.path.join(root, 'static')}"
        # os.popen("sudo -S %s" % command, 'w').write('Root2018!\n')
        image_names = os.listdir(os.path.join(root, 'static', dir_name))
        # return render_template('predict_result.html', dir_name=dir_name, image_names=image_names, predictions=output)
        uid = str(uuid.uuid4())
        PREDICTIONS[uid] = dict(dir_name=dir_name, image_names=image_names, predictions=output)
        return {'done': True, 'uid': uid}


@app.route('/prediction')
def prediction():
    uid = request.args.get('uid')
    data = PREDICTIONS.pop(uid)
    dir_name, image_names, predictions = data.get('dir_name'), data.get('image_names'), data.get('predictions')
    return render_template('predict_result.html', dir_name=dir_name, image_names=image_names, predictions=predictions)


if __name__ == '__main__':
    app.run(host='localhost', port=5005, debug=True)
