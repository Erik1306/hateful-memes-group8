# hateful-memes-group8
A project to classify hateful and non-hateful memes


### You can run this project using GUI or command line

1. Create virtual environment and install requirements:
```commandline
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. You should have the following files and folders to run train, test and predict.
- img.tar.gz is the directory of all the memes which are need for training, validation, and testing. Once extracted, images live in the img directory and have unique identifier ids as filenames, <id>.png
- train.jsonl is a .jsonl file, which is a list of json records, to be used for training. Each record had key-value pairs for an image id, filename img, extracted text from the image, and of course the image binary label. 0 is non-hateful and 1 is hateful.
- dev.jsonl provides the same keys, for the validation split.
- test.jsonl again has the same keys, except the label key.

# GUI instructions
3.1. To run project:
```commandline
python python3 web/server.py
```
3.2. Provide necessary files with GUI

# Command line instructions
1. 


