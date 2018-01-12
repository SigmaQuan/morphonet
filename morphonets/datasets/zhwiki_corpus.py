# -*- coding: utf-8 -*-
import os
import copy
import pickle
import random
# from keras.preprocessing.sequence import pad_sequences
import numpy as np
from collections import Counter
import sys
current_py_file = os.path.abspath(__file__)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(current_py_file))))

# import unicodedata
from morphonets.utils import pre_process_words
# from datasets import word_analogy_reasoning, word_similarity_computation
# from hist import hist
from morphonets.knowledge import cc_level_1
from morphonets.knowledge import cc_level_2
from morphonets.knowledge import cc_phonology
from morphonets.knowledge import cc_morphology
import morphonets.datasets.word_analogy_reasoning as war
import morphonets.datasets.word_similarity_computation as wsc
from keras.preprocessing.sequence import skipgrams
from configs.globals import PROJECT_FOLDER

ZHWIKI_FOLDER_CACHE = PROJECT_FOLDER + "/cache/corpus/"
ZHWIKI_FOLDER_DATA = PROJECT_FOLDER + "/data/corpus/"

CORPUS_NAME = "thulac_wiki_process.txt"


def skip_gram_generator(folder=ZHWIKI_FOLDER_CACHE, batch_size=5, context_window_size=5, negative_samples=5.):
    """Generate samples for training, testing, or validating language modeling
    on the Penn Treebank (PTB) dataset.

    # Arguments
        folder: the folder which contains all the corpora of training and
            testing and validation datasets of ptb.
        corpus_id: the id for denote train, test, or validation.
        batch_size: the size of one batch samples.
        context_window_size: size of context window.

    # Returns
        tuple of Numpy arrays:
        `(input_sequences, target_sequences)`.
    """
    # get the corpus_id_sequence of word id
    corpus_id_sequence = get_corpus_id_sequence(folder)
    # add by Robert Steven
    special_ids = get_special_ids()

    # get the size of vocabulary
    character_dic, id_character_dic = get_character_id_dictionaries()
    character_set = set(character_dic.keys())
    word_dic, id_word_dic = get_word_id_dictionaries()

    # get phonology knowledge
    character_pinyin_dic = cc_phonology.get_pinyins_of_character()
    fine_grained_pinyin_dic = cc_phonology.get_knowledge_of_fine_grained_pinyin()
    pinyin_id = cc_phonology.get_pinyin_id_dic()
    pinyin_set = set(pinyin_id.keys())
    id_initials, id_finals, id_tones = cc_phonology.get_id_other_dic()

    # get morphology knowledge

    # initialize index
    start_idx = 0

    num_classes_words = len(word_dic)
    # num_classes_characters = len(character_dic)
    # num_classes_pinyins = len(pinyin_id)
    # num_classes_initials = len(id_initials)
    # num_classes_id_finals = len(id_finals)
    # num_classes_tones = len(id_tones)

    # loop for generating batch of samples
    while True:
        # get a batch of samples
        input_sequence = generate_input_sequences(
            corpus_id_sequence, batch_size, start_idx, context_window_size)

        data, labels = skipgrams(sequence=input_sequence,
                                 vocabulary_size=num_classes_words,
                                 window_size=context_window_size,
                                 negative_samples=negative_samples)

        x = [np.array(x) for x in zip(*data)]
        y = np.array(labels, dtype=np.int32)

        # visualize_test_samples index
        # end_idx = start_idx + context_window_size
        # end_idx = start_idx + batch_size
        end_idx = start_idx + batch_size * (2 * context_window_size + 1)
        if end_idx > len(corpus_id_sequence):
            start_idx = end_idx - len(corpus_id_sequence)
        else:
            start_idx = end_idx

        # output a batch of samples
        yield (x, y)


def generator(folder=ZHWIKI_FOLDER_CACHE, batch_size=5, max_token_length=5):
    """Generate samples for training, testing, or validating language modeling
    on the Penn Treebank (PTB) dataset.

    # Arguments
        folder: the folder which contains all the corpora of training and
            testing and validation datasets of ptb.
        corpus_id: the id for denote train, test, or validation.
        batch_size: the size of one batch samples.
        max_token_length: length of corpus_id_sequence for each sample.

    # Returns
        tuple of Numpy arrays:
        `(input_sequences, target_sequences)`.
    """
    # get the corpus_id_sequence of word id
    corpus_id_sequence = get_corpus_id_sequence(folder)
    # add by Robert Steven
    special_ids = get_special_ids()

    # get the size of vocabulary
    character_dic, id_character_dic = get_character_id_dictionaries()
    character_set = set(character_dic.keys())
    word_dic, id_word_dic = get_word_id_dictionaries()

    # get phonology knowledge
    character_pinyin_dic = cc_phonology.get_pinyins_of_character()
    fine_grained_pinyin_dic = cc_phonology.get_knowledge_of_fine_grained_pinyin()
    pinyin_id = cc_phonology.get_pinyin_id_dic()
    pinyin_set = set(pinyin_id.keys())
    id_initials, id_finals, id_tones = cc_phonology.get_id_other_dic()

    # get morphology knowledge

    # initialize index
    start_idx = 0

    max_length = max_token_length * 4 + 1
    num_classes_words = len(word_dic)
    num_classes_characters = len(character_dic)
    num_classes_pinyins = len(pinyin_id)
    num_classes_initials = len(id_initials)
    num_classes_id_finals = len(id_finals)
    num_classes_tones = len(id_tones)

    # loop for generating batch of samples
    while True:
        # get a batch of samples
        # input_sequences = generate_batch_input_sequences(
        #     corpus_id_sequence, start_idx, batch_size,
        #     max_token_length)

        # get a batch of special samples
        input_sequences = generate_special_batch_input_sequences(
            corpus_id_sequence, start_idx, batch_size,
            max_token_length, special_ids)

        target_sequences_character, \
            target_sequences_pinyin, \
            target_sequences_initial, \
            target_sequences_final, \
            target_sequences_tone = generate_batch_target_sequences(
                input_sequences,
                character_set,
                character_dic, id_character_dic,
                word_dic, id_word_dic,
                pinyin_id, pinyin_set,
                character_pinyin_dic, fine_grained_pinyin_dic)

        # visualize_test_samples index
        end_idx = start_idx + 1
        if end_idx > len(corpus_id_sequence):
            start_idx = end_idx - len(corpus_id_sequence)
        else:
            start_idx = end_idx

        input_sequences = pad_sequences(input_sequences, maxlen=max_length, padding='app', truncating='app')
        target_sequences_character = pad_sequences(target_sequences_character, maxlen=max_length, padding='app', truncating='app')
        target_sequences_pinyin = pad_sequences(target_sequences_pinyin, maxlen=max_length, padding='app', truncating='app')
        target_sequences_initial = pad_sequences(target_sequences_initial, maxlen=max_length, padding='app', truncating='app')
        target_sequences_final = pad_sequences(target_sequences_final, maxlen=max_length, padding='app', truncating='app')
        target_sequences_tone = pad_sequences(target_sequences_tone, maxlen=max_length, padding='app', truncating='app')

        input_sequences = to_tensor(input_sequences, max_length)
        target_sequences_character = to_categorical(target_sequences_character, max_length, num_classes_characters)
        target_sequences_pinyin = to_categorical(target_sequences_pinyin, max_length, num_classes_pinyins)
        target_sequences_initial = to_categorical(target_sequences_initial, max_length, num_classes_initials)
        target_sequences_final = to_categorical(target_sequences_final, max_length, num_classes_id_finals)
        target_sequences_tone = to_categorical(target_sequences_tone, max_length, num_classes_tones)

        # output a batch of samples
        yield (input_sequences,
               [target_sequences_character,
                target_sequences_pinyin,
                target_sequences_initial,
                target_sequences_final,
                target_sequences_tone])


def get_corpus_id_sequence(folder=ZHWIKI_FOLDER_CACHE):
    """Get the formalized statistics of zhwiki.

    # Returns
        word_count: the frequency of each word.
        character_count: the frequency of each Chinese character.
        word_idx: the index of each word.
        character_idx: the index of each Chinese character.
    """
    dump_file = "zhwiki_corpus.id_sequence.pkl"
    dump_file = os.path.join(folder, dump_file)
    if os.path.exists(dump_file):
        # load from file
        with open(dump_file, 'rb') as f:
            id_sequences = pickle.load(f)
        return id_sequences
    else:
        pass

    corpus_file = os.path.join(ZHWIKI_FOLDER_DATA, CORPUS_NAME)
    assert (os.path.exists(corpus_file))
    corpus, tokens_number = read_corpus(corpus_file)

    # get the size of vocabulary
    character_dic, id_character_dic = get_character_id_dictionaries()
    character_set = set(character_dic.keys())
    word_dic, _ = get_word_id_dictionaries()
    word_set = set(word_dic.keys())

    id_sequences = []
    for i, token in enumerate(corpus):
        # print("{}, {}".format(i, token.encode('utf-8')))
        if token in word_set:
            if len(token) == 0:
                word_id = word_dic[u'SPECIAL_SYMBOL']
                id_sequences.append(word_id)
            else:
                id_sequences.append(word_dic[token])
        else:
            character_ids = word_to_character_ids(token, character_set, character_dic)
            for char in character_ids:
                print(id_character_dic[char])
            id_sequences += character_ids

    # save to file
    with open(dump_file, 'wb') as f:
        pickle.dump(id_sequences, f, protocol=2)

    return id_sequences


def generate_batch_input_sequences(
        corpus_id_sequence, start_idx, batch_size, max_token_length):
    """Generate a batch of input sequences or target sequences.

    # Arguments
        corpus_id_sequence: the word id corpus_id_sequence, which corresponds to a training
            corpus, or testing corpus, or validation corpus.
        start_idx: the index of the beginning of one batch.
        batch_size: the number of sequences in one batch samples.
        max_token_length: length of corpus_id_sequence for each sample.
        vocab_size: the size of vocabulary.

    # Returns
        sequences: a batch of sequences.
    """
    length = len(corpus_id_sequence)
    sequences = []
    for i in range(batch_size):
        token_length = random.randint(1, max_token_length)
        begin_idx = start_idx + i * token_length
        if begin_idx >= length:
            begin_idx -= length
        end_idx = begin_idx + token_length
        if end_idx < length:
            input_sequence = corpus_id_sequence[begin_idx: begin_idx + token_length]
            sequences.append(input_sequence)
        else:
            end_idx -= length
            input_sequence = corpus_id_sequence[begin_idx:]
            input_sequence.extend(corpus_id_sequence[:end_idx])
            sequences.append(input_sequence)

    # return sequences
    return sequences


def generate_input_sequences(
        corpus_id_sequence, batch_size, start_idx, context_window_size):
    """Generate a input ID sequence for skip gram.

    # Arguments
        corpus_id_sequence: the word id corpus_id_sequence, which corresponds to a training
            corpus, or testing corpus, or validation corpus.
        start_idx: the index of the beginning of one batch.
        batch_size: the number of sequences in one batch samples.
        context_window_size: length of corpus_id_sequence for each sample.
        vocab_size: the size of vocabulary.

    # Returns
        sequences: a batch of sequences.
    """
    length = len(corpus_id_sequence)

    token_length = 2 * context_window_size + 1
    # token_length = batch_size

    begin_idx = start_idx
    if begin_idx >= length:
        begin_idx -= length
    end_idx = begin_idx + batch_size * token_length
    # end_idx = begin_idx + token_length
    if end_idx < length:
        input_sequence = corpus_id_sequence[begin_idx: begin_idx + token_length]
    else:
        end_idx -= length
        input_sequence = corpus_id_sequence[begin_idx:]
        input_sequence.extend(corpus_id_sequence[:end_idx])

    # return sequences
    return input_sequence


def get_special_ids():
    word_dic, _ = get_word_id_dictionaries()
    word_id_set = set(word_dic.keys())

    words_s = wsc.get_total_words()
    words_a = war.get_total_words()
    # words = words_a.extend(words_s)

    word_ids = []
    for word in words_s:
        word_ = word.decode('utf-8')
        if word_ in word_id_set:
            word_ids.append(word_dic[word_])

    for word in words_a:
        word_ = word.decode('utf-8')
        if word_ in word_id_set:
            word_ids.append(word_dic[word_])

    word_ids = set(word_ids)

    return word_ids


def generate_special_batch_input_sequences(
        corpus_id_sequence, start_idx, batch_size, max_token_length, speical_id_sequence):
    """Generate a batch of input sequences or target sequences.

    # Arguments
        corpus_id_sequence: the word id corpus_id_sequence, which corresponds to a training
            corpus, or testing corpus, or validation corpus.
        start_idx: the index of the beginning of one batch.
        batch_size: the number of sequences in one batch samples.
        max_token_length: length of corpus_id_sequence for each sample.
        vocab_size: the size of vocabulary.

    # Returns
        sequences: a batch of sequences.
    """
    length = len(corpus_id_sequence)
    sequences = []

    if batch_size < max_token_length:
        batch_size = max_token_length

    for i in range(batch_size):
        while corpus_id_sequence[start_idx] not in speical_id_sequence:
            start_idx += 1
            if start_idx >= length:
                start_idx = 0

        token_length = random.randint(1, max_token_length)
        begin_idx = start_idx + random.randint(-token_length + 1, token_length)

        if begin_idx >= length:
            begin_idx -= length
        end_idx = begin_idx + token_length
        if end_idx < length:
            input_sequence = corpus_id_sequence[begin_idx: begin_idx + token_length]
            sequences.append(input_sequence)
        else:
            end_idx -= length
            input_sequence = corpus_id_sequence[begin_idx:]
            input_sequence.extend(corpus_id_sequence[:end_idx])
            sequences.append(input_sequence)

    # return sequences
    return sequences


def tokens_to_characters(token_sequence,
                         character_set, character_dic,
                         word_dic, id_word_dic,):
    character_sequence = []
    for i in range(len(token_sequence)):
        if token_sequence[i] <= word_dic[u'SPECIAL_SYMBOL']:
            character_sequence.append(token_sequence[i])
        else:
            character_sequence += word_to_character_ids(id_word_dic[token_sequence[i]],
                                                        character_set, character_dic)

    return character_sequence


def characters_to_phonologys(
        character_sequence,
        character_dic, id_character_dic,
        pinyin_id, pinyin_set,
        character_pinyin_dic, fine_grained_pinyin_dic):
    pinyin_sequence = []
    initial_sequence = []
    final_sequence = []
    tone_sequence = []
    for i in range(len(character_sequence)):
        character = id_character_dic[character_sequence[i]]
        if character == u'' or \
                        character == u'UN_KNOWN_CHAR' or \
                        character == u'SPECIAL_SYMBOL':
            pinyin = 'UN_KNOWN_CHAR'
            # initial_sequence.append(initial)
            # final_sequence.append(final)
            # tone_sequence.append(tone)
        else:
            pinyins = character_pinyin_dic[character]
            index = 0
            size = len(pinyins)
            if size > 1:
                index = random.randint(0, size - 1)
            pinyin = pinyins[index]

        pinyin_sequence.append(pinyin_id[pinyin])

        initial = fine_grained_pinyin_dic[pinyin]["Initial_ID"]
        initial_sequence.append(initial)

        final = fine_grained_pinyin_dic[pinyin]["Final_ID"]
        final_sequence.append(final)

        tone = fine_grained_pinyin_dic[pinyin]["Tone_ID"]
        tone_sequence.append(tone)

    return pinyin_sequence, initial_sequence, final_sequence, tone_sequence


def generate_batch_target_sequences(
        input_sequences,
        character_set,
        character_dic, id_character_dic,
        word_dic, id_word_dic,
        pinyin_id, pinyin_set,
        character_pinyin_dic, fine_grained_pinyin_dic):
    target_sequences_character = []
    target_sequences_pinyin = []
    target_sequences_initial = []
    target_sequences_final = []
    target_sequences_tone = []
    for i in range(len(input_sequences)):
        target_sequence_character = tokens_to_characters(
            input_sequences[i],
            character_set, character_dic,
            word_dic, id_word_dic,)

        target_sequence_pinyin, target_sequence_initial, \
            target_sequence_final, target_sequence_tone, = \
            characters_to_phonologys(target_sequence_character,
                                     character_dic, id_character_dic,
                                     pinyin_id, pinyin_set,
                                     character_pinyin_dic, fine_grained_pinyin_dic)

        max_length = len(input_sequences[i]) + len(target_sequence_character) + 1

        target_sequence_character = pad_sequence(
            target_sequence_character, maxlen=max_length, padding='pre', truncating='pre')
        target_sequence_pinyin = pad_sequence(
            target_sequence_pinyin, maxlen=max_length, padding='pre', truncating='pre')
        target_sequence_initial = pad_sequence(
            target_sequence_initial, maxlen=max_length, padding='pre', truncating='pre')
        target_sequence_final = pad_sequence(
            target_sequence_final, maxlen=max_length, padding='pre', truncating='pre')
        target_sequence_tone = pad_sequence(
            target_sequence_tone, maxlen=max_length, padding='pre', truncating='pre')

        target_sequences_character.append(target_sequence_character)
        target_sequences_pinyin.append(target_sequence_pinyin)
        target_sequences_initial.append(target_sequence_initial)
        target_sequences_final.append(target_sequence_final)
        target_sequences_tone.append(target_sequence_tone)

    return target_sequences_character, \
           target_sequences_pinyin, \
           target_sequences_initial, \
           target_sequences_final, \
           target_sequences_tone


def dump_unit_id_pairs(dump_file, unit_dic, id_unit_dic):
    """Save the file of the units of zhwiki corpora.

    # Arguments
        dump_file: the file path.
        unit_dic: the units of zhwiki corpora.
    """
    # print("dump ptb dictionary: ", dump_file)
    with open(dump_file, 'wb') as f:
        pickle.dump(unit_dic, f, protocol=2)
        pickle.dump(id_unit_dic, f, protocol=2)


def load_unit_id_pairs(dump_file):
    """Load the file of the units of zhwiki corpora.

    # Arguments
        dump_file: the file path.

    # Returns
        unit_dic: the units of zhwiki corpora.
    """
    # print("load ptb dictionary: %s " % dump_file)
    with open(dump_file, 'rb') as f:
        unit_dic = pickle.load(f)
        id_unit_dic = pickle.load(f)

    return unit_dic, id_unit_dic


def get_character_id_dictionaries(folder=ZHWIKI_FOLDER_CACHE):
    """Get the character id pairs of zhwiki.

    # Returns
        character_id_dic: the dictionary of character id pairs.
    """
    dump_file = "zhwiki_corpus.character_id_pairs.pkl"
    dump_file = os.path.join(folder, dump_file)
    if os.path.exists(dump_file):
        return load_unit_id_pairs(dump_file)
    else:
        pass

    _, character_count, _, _, _, _ = get_statistics(ZHWIKI_FOLDER_CACHE)
    cc_6500 = set(cc_level_1.get_characters() + cc_level_2.get_characters())

    character_id_dic = {}
    character_id_dic[u""] = 0
    character_id_dic[u'UN_KNOWN_CHAR'] = 1
    id = 1
    for item in character_count.items():
        character = item[0]
        if character in cc_6500 and item[1] >= 5:
            id += 1
            character_id_dic[character] = id
    character_id_dic[u'SPECIAL_SYMBOL'] = id

    id_character_dic = reverser(character_id_dic)

    dump_unit_id_pairs(dump_file, character_id_dic, id_character_dic)

    return character_id_dic, id_character_dic


def reverser(unit_id_dic):
    id_unit_dic = {}
    for item in unit_id_dic.items():
        id_unit_dic[item[1]] = item[0]

    return id_unit_dic


def word_to_character_ids(word, character_set, character_dic):
    characters = []
    if len(word) == 0:
        characters.append(character_dic[u'SPECIAL_SYMBOL'])
        return characters

    for character in word:
        if character not in character_set:
            characters.append(character_dic[u'UN_KNOWN_CHAR'])
        else:
            characters.append(character_dic[character])

    return characters


def get_word_id_dictionaries(folder=ZHWIKI_FOLDER_CACHE):
    """Get the word id pairs of zhwiki.

    # Returns
        word_id_dic: the dictionary of word id pairs.
    """
    dump_file = "zhwiki_corpus.word_id_pairs.pkl"
    dump_file = os.path.join(folder, dump_file)
    if os.path.exists(dump_file):
        return load_unit_id_pairs(dump_file)
    else:
        pass

    character_id_dic, _ = get_character_id_dictionaries(folder)
    word_count, _, _, _, _, _ = get_statistics(ZHWIKI_FOLDER_CACHE)

    id = len(character_id_dic) - 1
    word_id_dic = copy.deepcopy(character_id_dic)
    character_set = set(character_id_dic.keys())
    for item in word_count.items():
        word = item[0]
        # print(word)

        # if len(word) == 0:
        #     id += 1
        #     word_id_dic[u'SPECIAL_SYMBOL'] = id
        if item[1] >= 5 and len(word) > 1:
            flag = True
            for character in word:
                if character not in character_set:
                    flag = False
                    break
            if flag is True:
                id += 1
                word_id_dic[word] = id

    id_word_dic = reverser(word_id_dic)

    dump_unit_id_pairs(dump_file, word_id_dic, id_word_dic)

    return word_id_dic, id_word_dic


# delete
# def get_formalized_statistics(folder=ZHWIKI_FOLDER_CACHE):
#     """Get the formalized statistics of zhwiki.
#
#     # Returns
#         word_count: the frequency of each word.
#         character_count: the frequency of each Chinese character.
#         word_idx: the index of each word.
#         character_idx: the index of each Chinese character.
#     """
#     dump_file = "zhwiki_corpus.formalized_statistics.pkl"
#     dump_file = os.path.join(folder, dump_file)
#     # if os.path.exists(dump_file):
#     #     return load_statistics(dump_file)
#     # else:
#     #     pass
#
#     word_count, character_count, word_idx, character_idx, tokens_number, \
#         characters_number = get_statistics(ZHWIKI_FOLDER_CACHE)
#
#     totals_w = sum(word_count.values())
#     word_count_list = sorted(word_count.items(),
#                              key=lambda d: d[1],
#                              everse=True)
#     word_distribution = []
#     count = 0
#     for item in word_count_list:
#         count += item[1]
#         word_distribution.append(count / float(totals_w))
#
#     totals_c = sum(character_count.values())
#     character_count_list = sorted(character_count.items(),
#                                   key=lambda d: d[1],
#                                   reverse=True)
#     character_distribution = []
#     count = 0
#     for item in character_count_list:
#         count += item[1]
#         character_distribution.append(count / float(totals_c))
#
#     total_words_1 = word_similarity_computation.get_total_words()
#     total_characters_1 = word_similarity_computation.get_total_characters(total_words_1)
#     total_words_2 = word_analogy_reasoning.get_total_words()
#     total_characters_2 = word_similarity_computation.get_total_characters(total_words_2)
#
#     total_words = set(total_words_1) | set(total_words_2)
#     total_characters = set(total_characters_1) | set(total_characters_2)
#
#     i = 0
#     count = 0
#     print(len(total_words))
#     for item in total_words:
#         if item in word_count.keys():
#             i += 1
#             count += word_count[item]
#             print("{:>11}, {:>8}, {:<15}, {}".format(
#                 i, word_count[item], count / float(totals_w), item))
#         else:
#             print("******: {}".format(item))
#
#     i = 0
#     count = 0
#     print(len(total_characters))
#     for item in total_characters:
#         if item in character_count.keys():
#             i += 1
#             count += character_count[item]
#             print("{:>11}, {:>8}, {:<15}, {}".format(
#                 i, character_count[item], count / float(totals_c), item))
#         else:
#             print("******: {}".format(item))
#
#     return word_count, character_count, word_idx, character_idx, tokens_number, characters_number, stop_words


def get_statistics(folder=ZHWIKI_FOLDER_CACHE):
    """Get the statistics of zhwiki.

    # Arguments
        folder: the folder which contains the zhwiki corpora.

    # Returns
        word_count: the frequency of each word.
        character_count: the frequency of each Chinese character.
        word_idx: the index of each word.
        character_idx: the index of each Chinese character.
    """
    dump_file = "zhwiki_corpus.statistics.pkl"
    dump_file = os.path.join(folder, dump_file)
    if os.path.exists(dump_file):
        return load_statistics(dump_file)
    else:
        pass

    # print('\n-')
    # print "Get file path from folder: %s" % folder
    corpus_file = os.path.join(ZHWIKI_FOLDER_DATA, CORPUS_NAME)
    assert (os.path.exists(corpus_file))

    # print('\n-')
    print("Read file: %s" % corpus_file)
    corpus, tokens_number = read_corpus(corpus_file)
    character_corpus = u"".join(corpus)
    characters_number = len(character_corpus)

    # get frequency of each word and each character
    word_count = Counter(corpus)
    character_count = Counter(character_corpus)

    # vocab = sorted(Counter(corpus).keys())
    word = sorted(word_count.keys())
    character = sorted(character_count.keys())

    # Reserve 0 for masking via pad_sequences
    # word_size = len(word) + 1
    # character_size = len(character) + 1

    # print('-')
    # print('Vectorizing the word sequences...')
    word_idx = dict((w, i + 1) for i, w in enumerate(word))
    character_idx = dict((c, i + 1) for i, c in enumerate(character))

    # save statistics to file
    dump_statistics(dump_file, word_count, character_count, word_idx,
                    character_idx, tokens_number, characters_number)

    return word_count, character_count, word_idx, character_idx, tokens_number, characters_number


def dump_statistics(dump_file, word_count, character_count, word_idx,
                    character_idx, tokens_number, characters_number):
    """Save the statistics information about the zhwiki corpora.

    # Arguments
        dump_file: the file path.
        word_count: the frequency of each word.
        character_count: the frequency of each Chinese character.
        word_idx: the index of each word.
        character_idx: the index of each Chinese character.
    """
    # print("dump ptb dictionary: ", dump_file)
    with open(dump_file, 'wb') as f:
        pickle.dump(word_count, f, protocol=2)
        pickle.dump(character_count, f, protocol=2)
        pickle.dump(word_idx, f, protocol=2)
        pickle.dump(character_idx, f, protocol=2)
        pickle.dump(tokens_number, f, protocol=2)
        pickle.dump(characters_number, f, protocol=2)


def load_statistics(dump_file):
    """Load the statistics information about the zhwiki corpora.

    # Arguments
        dump_file: the file path.

    # Returns
        word_count: the frequency of each word.
        character_count: the frequency of each Chinese character.
        word_idx: the index of each word.
        character_idx: the index of each Chinese character.
    """
    # print("load ptb dictionary: %s " % dump_file)
    with open(dump_file, 'rb') as f:
        word_count = pickle.load(f)
        character_count = pickle.load(f)
        word_idx = pickle.load(f)
        character_idx = pickle.load(f)
        tokens_number = pickle.load(f)
        characters_number = pickle.load(f)

    return word_count, character_count, word_idx, character_idx, tokens_number, characters_number
#
#
# def get_dictionary(folder=ZHWIKI_FOLDER_CACHE):
#     """Loads the text8 dictionary.
#
#     # Arguments
#         folder: the folder which contains all the corpora of training and
#             development datasets of text8.
#
#     # Returns
#         vocab: total vocabulary of the corpora.
#         vocab_size: the size of vocabulary.
#         word_idx: the index of each word.
#     """
#     # get the sequence of word id
#     dump_file = "zhwiki_corpus.dictionary.pkl"
#     dump_file = os.path.join(folder, dump_file)
#     # if os.path.exists(dump_file):
#     #     return load_dict(dump_file)
#     # else:
#     #     pass
#
#     # print('\n-')
#     # print "Get file path from folder: %s" % folder
#     corpus_file = os.path.join(ZHWIKI_FOLDER_DATA, CORPUS_NAME)
#     assert (os.path.exists(corpus_file))
#
#     print("Read file: %s" % corpus_file)
#     corpus = read_corpus(corpus_file)
#
#     vocab = sorted(Counter(corpus).keys())
#
#     # Reserve 0 for masking via pad_sequences
#     vocab_size = len(vocab) + 1
#
#     # print('-')
#     # print('Vectorizing the word sequences...')
#     word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
#
#     # save_sample_image to file
#     dump_dict(dump_file, vocab, vocab_size, word_idx)
#
#     return vocab, vocab_size, word_idx
#
#
# def dump_dict(dump_file, vocab, vocab_size, word_idx):
#     """Save the information of dictionary about text8.
#
#     # Arguments
#         dump_file: the for save_sample_image information.
#         vocab: the total words in text8.
#         vocab_size: the size of vocabulary.
#         word_idx: the index of each word.
#     """
#     # print("dump ptb dictionary: ", dump_file)
#     with open(dump_file, 'wb') as f:
#         pickle.dump(vocab, f, protocol=2)
#         pickle.dump(vocab_size, f, protocol=2)
#         pickle.dump(word_idx, f, protocol=2)
#
#
# def load_dict(dump_file):
#     """Load the information of dictionary about text8.
#
#     # Arguments
#         dump_file: the for save_sample_image information.
#
#     # Returns
#         vocab: the total words in text8.
#         vocab_size: the size of vocabulary.
#         word_idx: the index of each word.
#     """
#     # print("load ptb dictionary: %s " % dump_file)
#     with open(dump_file, 'rb') as f:
#         vocab = pickle.load(f)
#         vocab_size = pickle.load(f)
#         word_idx = pickle.load(f)
#
#     return vocab, vocab_size, word_idx


def read_corpus(file_path, folder=ZHWIKI_FOLDER_CACHE):
    """Get all lines of data from one corpus.

    # Arguments
        file_path: the folder which contains all the corpora for training
                   embeddings.

    # Returns
        a group of token sequences.
    """
    dump_file = "zhwiki_corpus.my.dump.pkl"
    dump_file = os.path.join(folder, dump_file)
    if os.path.exists(dump_file):
        return load_zhwiki_corpus(dump_file)
    else:
        pass

    corpus = []
    tokens_number = 0
    for i, line in enumerate(open(file_path)):
        line = line.decode('utf-8').strip()
        one_line = line.split(u' ')
        one_line = pre_process_words(one_line)
        one_line.append(u'')
        tokens_number += len(one_line)
        corpus.extend(one_line)

        # for debug code
        # if i > 1000:
        #     break

    dump_zhwiki_corpus(dump_file, corpus, tokens_number)

    return corpus, tokens_number


def dump_zhwiki_corpus(dump_file, corpus, tokens_number):
    """Save the file of the zhwiki corpora.

    # Arguments
        dump_file: the file path.
        corpus: the zhwiki corpora.
        tokens_number: the number of tokens.
    """
    # print("dump ptb dictionary: ", dump_file)
    with open(dump_file, 'wb') as f:
        pickle.dump(corpus, f, protocol=2)
        pickle.dump(tokens_number, f, protocol=2)


def load_zhwiki_corpus(dump_file):
    """Load the file of the zhwiki corpora.

    # Arguments
        dump_file: the file path.

    # Returns
        corpus: the zhwiki corpora.
        tokens_number: the number of tokens.
    """
    # print("load ptb dictionary: %s " % dump_file)
    with open(dump_file, 'rb') as f:
        corpus = pickle.load(f)
        tokens_number = pickle.load(f)

    return corpus, tokens_number


def to_tensor(sequences, sample_sequence_length):
    """Converts a group of a class vector (integers) to a group binary class matrix.

    E.g. for use with categorical_crossentropy.

    # Arguments
        id_sequence: class id vector to be converted into a matrix
            (integers from 0 to num_classes).

    # Returns
        A binary matrix representation of the input.
    """
    sample_number = len(sequences)
    tensor = np.zeros(shape=(sample_number,
                             sample_sequence_length),
                      dtype=np.uint64)

    for i in range(sample_number):
        for j in range(sample_sequence_length):
            tensor[i, j] = sequences[i][j]

    return tensor


def to_categorical(sequences, sample_sequence_length, num_classes):
    """Converts a group of a class vector (integers) to a group binary class matrix.

    E.g. for use with categorical_crossentropy.

    # Arguments
        id_sequence: class id vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.

    # Returns
        A binary matrix representation of the input.
    """
    sample_number = len(sequences)
    tensor = np.zeros(shape=(sample_number,
                             sample_sequence_length,
                             num_classes),
                      dtype=np.uint8)

    for i in range(sample_number):
        for j in range(sample_sequence_length):
            if sequences[i][j] > 0:
                tensor[i, j, sequences[i][j]] = 1

    return tensor


def pad_sequence(sequence, maxlen, dtype='int32',
                  padding='pre', truncating='pre', value=0.0):

    paded_seqence = np.zeros(maxlen, dtype=np.uint32)

    if len(sequence) == maxlen:
        return sequence

    elif len(sequence) < maxlen:
        if padding == 'pre':
            j = maxlen - 1
            for i in range(len(sequence)-1, -1, -1):
                paded_seqence[j] = sequence[i]
                j -= 1
        else:
            for i in range(len(sequence)):
                paded_seqence[i] = sequence[i]
    else:
        if truncating == 'pre':
            j = maxlen - 1
            for i in range(maxlen-1, -1, -1):
                paded_seqence[j] = sequence[i]
                j -= 1
        else:
            for i in range(len(paded_seqence)):
                paded_seqence[i] = sequence[i]

    return paded_seqence


def pad_sequences(sequences, maxlen,
                  padding, truncating, value=0):
    paded_sequences = []
    for i in range(len(sequences)):
        paded_sequences.append(pad_sequence(
            sequences[i], maxlen=maxlen, padding=padding, truncating=truncating))

    return paded_sequences


def decode_ids_to_symbols(
        id_word_dic, id_pinyins, id_initials, id_finals, id_tones,
        input_sequence, target_sequence):
    targets_character, target_pinyin, target_initial, target_final, target_tone = target_sequence
    inputs = u""
    for k in range(len(input_sequence)):
        inputs += u"{}".format(id_word_dic[input_sequence[k]])
        # print(id_word_dic[input_sequence[k]])

    print('targets_character')
    print(targets_character)
    targets_c = u""
    for k in range(len(targets_character)):
        targets_c += u"{}".format(id_word_dic[np.argmax(targets_character[k])])
        # print(id_word_dic[np.argmax(targets_character[k])])

    print('target_pinyin')
    print(target_pinyin)
    targets_p = u""
    for k in range(len(target_pinyin)):
        targets_p += u"{}".format(id_pinyins[np.argmax(target_pinyin[k])])
        # print(id_pinyins[np.argmax(target_pinyin[k])])

    print('target_initial')
    print(target_initial)
    targets_i = u""
    for k in range(len(target_initial)):
        targets_i += u"{}".format(id_initials[np.argmax(target_initial[k])])
        # print(id_initials[np.argmax(target_initial[k])])

    print('target_final')
    print(target_final)
    targets_f = u""
    for k in range(len(target_final)):
        targets_f += u"{}".format(id_finals[np.argmax(target_final[k])])
        # print(id_finals[np.argmax(target_final[k])])

    print('target_tone')
    print(target_tone)
    targets_t = u""
    for k in range(len(target_tone)):
        targets_t += u"{}".format(id_tones[np.argmax(target_tone[k])])
        # print(id_tones[np.argmax(target_tone[k])])

    return inputs, [targets_c, targets_p, targets_i, targets_f, targets_t]
