# -*- coding: utf-8 -*-
import argparse
import time
import json
import os
import sys

current_py_file = os.path.abspath(__file__)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(current_py_file))))

from configs.globals import PROJECT_FOLDER
from morphonets.datasets import zhwiki_corpus
from morphonets.knowledge import cc_phonology


SKIP_GRAM_FOLDER = PROJECT_FOLDER + '/logs/mlp_skip_gram/'
SKIP_GRAM_MODEL_FOLDER = PROJECT_FOLDER + '/data/model_file/mlp_skip_gram/'
FOLDER = SKIP_GRAM_FOLDER + time.strftime('%Y-%m-%d-%H-%M-%S/')


if not os.path.isdir(FOLDER):
    os.makedirs(FOLDER)
    print(time.strftime('%Y-%m-%d %H:%M:%S') + " Create folder: %s" % FOLDER)
if not os.path.isdir(SKIP_GRAM_MODEL_FOLDER):
    os.makedirs(SKIP_GRAM_MODEL_FOLDER)
    print(time.strftime('%Y-%m-%d %H:%M:%S') + " Create folder: %s" % SKIP_GRAM_MODEL_FOLDER)


def get_args():
    parser = argparse.ArgumentParser()

    # corpus parameter
    word_average_length = 3
    context_window_size = 5
    parser.add_argument('--context_window_size', type=int, default=context_window_size,
                        help='the size of context window')

    parser.add_argument('--embedding_size', type=int, default=100,
                        help='the size of embeddings')

    _, _, _, _, tokens_number, _ = zhwiki_corpus.get_statistics()
    parser.add_argument('--corpus_size', type=int, default=tokens_number,
                        help='the number of word in train corpus')

    word_dic, _ = zhwiki_corpus.get_word_id_dictionaries()
    vocabulary_size = len(word_dic)
    parser.add_argument('--vocabulary_size', type=int, default=vocabulary_size,
                        help='the size of vocabulary')

    character_dic, _ = zhwiki_corpus.get_character_id_dictionaries()
    character_size = len(character_dic)
    parser.add_argument('--character_size', type=int, default=character_size,
                        help='the size of character')

    pinyin_id = cc_phonology.get_pinyin_id_dic()
    pinyin_size = len(pinyin_id)
    parser.add_argument('--pinyin_size', type=int, default=pinyin_size,
                        help='the size of pinyin')

    id_initials, id_finals, id_tones = cc_phonology.get_id_other_dic()
    initial_size, final_size, tone_size = len(id_initials), len(id_finals), len(id_tones)
    parser.add_argument('--initial_size', type=int, default=initial_size,
                        help='the size of initial')
    parser.add_argument('--final_size', type=int, default=final_size,
                        help='the size of initial')
    parser.add_argument('--tone_size', type=int, default=tone_size,
                        help='the size of initial')

    # model training parameter
    # batch_size = 5000
    batch_size = 2000
    # batch_size = 5
    parser.add_argument('--batch_size', type=int, default=batch_size,
                        help='size of batch sample')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate (between 0 and 1)')
    epochs = 50
    # epochs = 5
    parser.add_argument('--epochs', type=int, default=epochs,
                        help='number of epochs')
    parser.add_argument("--gpus", type=int, default=2,
                        help="# of GPUs to use for training")

    steps_per_epoch = tokens_number / (batch_size * (2 * context_window_size + 1))
    # steps_per_epoch *= 5
    steps_per_epoch *= 2
    # steps_per_epoch = tokens_number / batch_size
    # steps_per_epoch = tokens_number / context_window_size / 100
    # steps_per_epoch = 10
    parser.add_argument('--steps_per_epoch', type=int, default=steps_per_epoch,
                        help='the number of batches for training')
    iterations = 20 * 200
    parser.add_argument('--iterations', type=int, default=iterations,
                        help='number of iterations')

    # model saving parametsder
    model_file_train = SKIP_GRAM_MODEL_FOLDER + "model_train_dimension_100.hdf5"
    parser.add_argument('--model_file_train', type=str, default=model_file_train,
                        help='load model from a file')
    model_file_test = SKIP_GRAM_MODEL_FOLDER + "model_test.hdf5"
    parser.add_argument('--model_file_test', type=str, default=model_file_test,
                        help='load model from a file for testing')
    model_picture = FOLDER + "architecture.png"
    parser.add_argument('--model_picture', type=str, default=model_picture,
                        help='the file for saving the model architecture')
    parser.add_argument('--log_every', type=int, default=1,
                        help='print information every x iteration')
    parser.add_argument('--save_every', type=int, default=1,
                        help='save_sample_image state every x epoch')

    args = parser.parse_args()
    return args


def get(file_path=None):
    if file_path is None:
        file_path = FOLDER + "configuration.json"

    args = get_args()

    argument_list = vars(args)
    # print(type(vars(args)))
    configs = json.dumps(argument_list, indent=1)
    print(time.strftime('%Y-%m-%d %H:%M:%S') + " Model configuration: %s" % configs)

    with open(file_path, 'w') as json_file:
        json_file.write(configs)

    return args
