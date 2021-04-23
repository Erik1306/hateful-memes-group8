from flask import Flask, render_template, request
from pathlib import Path
import zipfile
from model import HatefulMemesModel

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/train')
def train():
    return render_template('train.html')


@app.route('/test')
def test():
    return render_template('test.html')


@app.route('/predict')
def predict():
    return render_template('predict.html')


@app.route('/train_result')
def train_result():
    base_dir = request.args.get('images')
    images = request.args.get('images')
    train_path = request.args.get('train')
    test_path = request.args.get('test')
    dev_path = request.args.get('dev')

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
        "output_path": "model-outputs",
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
    return render_template('train_result.html')


@app.route('/test_result')
def test_result():
    images = request.args.get('images')
    test_path = request.args.get('test')
    return render_template('test_result.html')


@app.route('/predict_result')
def predict_result():
    images = request.args.get('images')
    return render_template('predict_result.html')


if __name__ == '__main__':
    app.run(host='localhost', port=5005, debug=True)
