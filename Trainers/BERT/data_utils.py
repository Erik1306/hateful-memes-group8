import tokenization
import re
import keras.backend as K
import numpy as np
import keras
import random
import os

np.random.seed(112)
random.seed(112)

BERT_URL = 'https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip'


class InputExample(object):
    def __init__(self, unique_id, text_a, text_b):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b


class InputFeatures(object):
    def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids


def read_examples(input_file):
    examples = []
    unique_id = 0
    with open(input_file, "r") as reader:
        while True:
            line = tokenization.convert_to_unicode(reader.readline())
            if not line:
                break
            line = line.strip()
            text_a = None
            text_b = None
            m = re.match(r"^(.*) \|\|\| (.*)$", line)
            if m is None:
                text_a = line
            else:
                text_a = m.group(1)
                text_b = m.group(2)
            examples.append(
                InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b))
            unique_id += 1
    return examples


def read_examples_list(input_list):
    examples = []
    unique_id = 0
    for line in input_list:
        line_new = tokenization.convert_to_unicode(line)
        if not line:
            print("Breaking for empty line !")
            break
        else:
            line = line_new
        line = line.strip()
        text_a = None
        text_b = None
        m = re.match(r"^(.*) \|\|\| (.*)$", line)
        if m is None:
            text_a = line
        else:
            text_a = m.group(1)
            text_b = m.group(2)
        examples.append(
            InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b))
        unique_id += 1
    return examples


def to_features(examples, seq_length, tokenizer, Verbose=False):
    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
        else:
            if len(tokens_a) > seq_length - 2:
                tokens_a = tokens_a[0:(seq_length - 2)]
        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                input_type_ids.append(1)
            tokens.append("[SEP]")
            input_type_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length

        if ex_index < 5 and Verbose:
            print("*** Example ***")
            print("unique_id: %s" % (example.unique_id))
            print("tokens: %s" % " ".join([str(x) for x in tokens]))
            print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            print("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            print(
                "input_type_ids: %s" % " ".join([str(x) for x in input_type_ids]))

        features.append(
            InputFeatures(
                unique_id=example.unique_id,
                tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def prepare_bert_tokens(input_file, vocab_path, max_seq_length):
    examples = read_examples(input_file)

    tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab_path, do_lower_case=True)

    features = to_features(
        examples=examples, seq_length=max_seq_length, tokenizer=tokenizer)

    unique_id_to_feature = {}
    for feature in features:
        unique_id_to_feature[feature.unique_id] = feature

    position_ids = list(np.arange(max_seq_length))
    all_input_ids = [[f.input_ids, f.input_mask, f.input_type_ids, position_ids] for f in features]

    return np.array(all_input_ids)


def get_bert_embeddings(model_pre_trained, input_file, max_seq_length=128):
    x = prepare_bert_tokens(input_file, model_pre_trained.vocab_path, max_seq_length=max_seq_length)

    bert_embedding = model_pre_trained(K.variable(x))

    return bert_embedding


class BertTokenizer(object):
    def __init__(self, model_dir=None, bert_url=BERT_URL,
                 input_file=None, input_list=[], max_seq_length=128):

        self.model_dir = model_dir
        if self.model_dir is not None and not os.path.exists(self.model_dir):
            raise ValueError(
                'Please provide a valid path to BERT pre-trained weights or set the value to None to automatically download the model weights')

        elif self.model_dir is None:

            folder_name = 'pre_trained_models/'
            model_name = bert_url.split('/')[-1]
            cmd = "mkdir " + folder_name + "  ; " + \
                  "wget " + bert_url + " -P " + folder_name + "  ;" + \
                  "unzip " + folder_name + model_name + " -d " + folder_name

            self.model_dir = folder_name + model_name.split('.')[0] + '/'
            if not os.path.exists(self.model_dir):
                print("Downloading pre-trained weights...")
                os.system(cmd)

        self.max_seq_length = max_seq_length
        if input_file is not None:
            self.examples = read_examples(input_file)
        else:
            self.examples = read_examples_list(input_list)
        self.tokenizer = tokenization.FullTokenizer(vocab_file=self.model_dir + 'vocab.txt', do_lower_case=True)
        self.features = to_features(examples=self.examples, seq_length=max_seq_length, tokenizer=self.tokenizer)
        self.unique_id_to_feature = {}
        for feature in self.features:
            self.unique_id_to_feature[feature.unique_id] = feature
        self.position_ids = list(np.arange(max_seq_length))
        self.all_input_ids = np.array(
            [[f.input_ids, f.input_mask, f.input_type_ids, self.position_ids] for f in self.features])


class DataGenerator(keras.utils.Sequence):
    def __init__(self, data, labels, batch_size=32, n_channels=1,
                 n_classes=2, shuffle=True):
        self.batch_size = batch_size
        self.labels = labels
        self.data = data
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.sample_size = self.data.shape[0]

        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(self.sample_size / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.data[indexes], self.labels[indexes]

        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(self.sample_size)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
