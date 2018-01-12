# -*- coding: utf-8 -*-
from keras.layers import Input, Embedding, LSTM, Dense, concatenate
from keras.utils.vis_utils import plot_model
from keras.optimizers import Adam
from keras.models import Model
import time
# from datasets import zhwiki_corpus
# from knowledge import cc_phonology
# from keras.callbacks import ModelCheckpoint
# import os

#
# import configs
#
#
# SKIP_GRAM_FOLDER = os.path.dirname(os.path.dirname(
#     os.path.abspath(__file__))) + '/cache/morphonet1/'
# SKIP_GRAM_MODEL_FOLDER = os.path.dirname(os.path.dirname(
#     os.path.abspath(__file__))) + '/data/model_file/morphonet1/'
# FOLDER = SKIP_GRAM_FOLDER + time.strftime('%Y-%m-%d-%H-%M-%S/')
#
#
# # get the size of vocabulary
# character_dic, id_character_dic = zhwiki_corpus.get_character_id_dictionaries()
# character_set = set(character_dic.keys())
# word_dic, id_word_dic = zhwiki_corpus.get_word_id_dictionaries()
#
#
# # get phonology knowledge
# character_pinyin_dic = cc_phonology.get_pinyins_of_character()
# fine_grained_pinyin_dic = cc_phonology.get_knowledge_of_fine_grained_pinyin()
# pinyin_id = cc_phonology.get_pinyin_id_dic()
# pinyin_set = set(pinyin_id.keys())
# id_initials, id_finals, id_tones = cc_phonology.get_id_other_dic()
#
# # get morphology knowledge
# num_classes_words = len(word_dic)
# num_classes_characters = len(character_dic)
# num_classes_pinyins = len(pinyin_id)
# num_classes_initials = len(id_initials)
# num_classes_id_finals = len(id_finals)
# num_classes_tones = len(id_tones)
#
# _, _, _, _, tokens_number, _ = zhwiki_corpus.get_statistics()
#
# BATCH_SIZE = 16
# MAX_TOKEN_LENGTH = 5
# WORD_AVERAGE_LENGTH = 3
# MAX_LENGTH = MAX_TOKEN_LENGTH * (WORD_AVERAGE_LENGTH + 1) + 1
#
# EMBEDDING_SIZE = 100
# HIDDEN_SIZE = EMBEDDING_SIZE * 3
# LEARNING_RATE = 0.0001
# # create folder
# if not os.path.isdir(FOLDER):
#     os.makedirs(FOLDER)
#     print(time.strftime('%Y-%m-%d %H:%M:%S') + " Create folder: %s" % FOLDER)
# if not os.path.isdir(SKIP_GRAM_MODEL_FOLDER):
#     os.makedirs(SKIP_GRAM_MODEL_FOLDER)
#     print(time.strftime('%Y-%m-%d %H:%M:%S') + " Create folder: %s" % SKIP_GRAM_MODEL_FOLDER)
#
# # create log file
# start_time = time.time()
#
#
# # Headline input: meant to receive sequences of 100 integers, between 1 and 10000.
# # Note that we can name any layer by passing it a "name" argument.
# input = Input(shape=(MAX_LENGTH,), dtype='int32', name='input')
#
# # This embedding layer will encode the input sequence
# # into a sequence of dense 512-dimensional vectors.
# x = Embedding(output_dim=EMBEDDING_SIZE,
#               input_dim=num_classes_words,
#               input_length=MAX_LENGTH,)(input)
#
# # A LSTM will transform the vector sequence into a single vector,
# # containing information about the entire sequence
# h_lstm_out = LSTM(HIDDEN_SIZE,
#               return_sequences=True,
#               batch_input_shape=(BATCH_SIZE, MAX_LENGTH, HIDDEN_SIZE))(x)
#
# h_initial_1 = Dense(EMBEDDING_SIZE, activation='softmax')(h_lstm_out)
# initial = Dense(num_classes_initials, activation='softmax', name='initial')(h_initial_1)
#
# h_final_1 = Dense(EMBEDDING_SIZE, activation='softmax')(h_lstm_out)
# final = Dense(num_classes_id_finals, activation='softmax', name='final')(h_final_1)
#
# h_tone_1 = Dense(EMBEDDING_SIZE, activation='sigmoid')(h_lstm_out)
# tone = Dense(num_classes_tones, activation='softmax', name='tone')(h_tone_1)
#
# h_pinyin_1 = concatenate([h_initial_1, h_final_1, h_tone_1])
# h_pinyin_2 = Dense(EMBEDDING_SIZE, activation='sigmoid')(h_pinyin_1)
# pinyin = Dense(num_classes_pinyins, activation='softmax', name='pinyin')(h_pinyin_2)
#
# h_out = concatenate([h_lstm_out, h_pinyin_2])
# h_out_1 = Dense(EMBEDDING_SIZE, activation='sigmoid')(h_out)
# character = Dense(num_classes_characters, activation='softmax', name='character')(h_out_1)
#
# model = Model(inputs=input,
#               outputs=[character,
#                        pinyin,
#                        initial,
#                        final,
#                        tone])
#
# # initialize the optimizer
# ADAM_ = Adam(lr=LEARNING_RATE)
#
# # compile the model
# model.compile(loss={'character': 'categorical_crossentropy',
#                     'pinyin': 'categorical_crossentropy',
#                     'initial': 'categorical_crossentropy',
#                     'final': 'categorical_crossentropy',
#                     'tone': 'categorical_crossentropy'},
#               loss_weights={'character': 1,
#                             'pinyin': 1,
#                             'initial': 0.2,
#                             'final': 0.2,
#                             'tone': 0.2},
#               optimizer=ADAM_,
#               metrics=['categorical_accuracy'])
#
# # save_sample_image the picture of the model
# print(time.strftime('%Y-%m-%d %H:%M:%S') +
#       ' Save picture of model architecture')
# plot_model(model, show_shapes=True,
#            to_file=FOLDER+"architecture_lstm_lm_ptb.png")
#
# # show the information of the model
# print(time.strftime('%Y-%m-%d %H:%M:%S') + " Model summary")
# print(model.summary())
#
# # begin training
# print(time.strftime('%Y-%m-%d %H:%M:%S') + " Begin training..")
# # Train the model each generation and show predictions against the
# # validation dataset
# iterations = 1000
# epochs = 10
# steps_per_epoch_ = tokens_number / (BATCH_SIZE * MAX_TOKEN_LENGTH) / 100
#
# model_file = "morphonet_1.hdf5"
# save_best_model = ModelCheckpoint(
#     filepath=FOLDER+model_file,
#     verbose=0, save_best_only=True)
#
# for iteration in range(1, iterations):
#     print(time.strftime('%Y-%m-%d %H:%M:%S') + ' Iteration %d ' % iteration)
#     model.fit_generator(zhwiki_corpus.generator(batch_size=BATCH_SIZE,
#                                                 max_token_length=MAX_TOKEN_LENGTH),
#                         steps_per_epoch=steps_per_epoch_,
#                         epochs=epochs,
#                         callbacks=[save_best_model],
#                         validation_data=zhwiki_corpus.generator(
#                             batch_size=BATCH_SIZE,
#                             max_token_length=MAX_TOKEN_LENGTH),
#                         validation_steps=10,
#                         verbose=0)


def get_embedding_layer(model):
    # for i in range(len(model.layers)):
    #     print("layer {}".format(i))
    #     weights = model.layers[i].get_weights()
    #     for j in range(len(weights)):
    #         print("weight {}, {}".format(i, j))
    #         print(weights[j])

    weights = model.layers[1].get_weights()[0]
    return weights


def get(args):
    # get figuration

    # Headline input: meant to receive sequences of 21 integers, between 1 and args.episode_size.
    # Note that we can name any layer by passing it a "name" argument.
    input = Input(shape=(args.episode_size,), dtype='int32', name='input')

    # This embedding layer will encode the input sequence
    # into a sequence of dense 512-dimensional vectors.
    x = Embedding(output_dim=args.embedding_size,
                  input_dim=args.vocabulary_size,
                  input_length=args.episode_size, )(input)

    # A LSTM will transform the vector sequence into a single vector,
    # containing information about the entire sequence
    h_lstm_out = LSTM(args.hidden_size, return_sequences=True,
                      batch_input_shape=(args.batch_size, args.episode_size, args.hidden_size))(x)

    # h_initial_1 = Dense(args.embedding_size, activation='softmax')(h_lstm_out)
    h_initial_1 = LSTM(args.embedding_size, return_sequences=True,
                       batch_input_shape=(args.batch_size, args.episode_size, args.embedding_size))(h_lstm_out)
    initial = Dense(args.initial_size, activation='softmax', name='initial')(h_initial_1)

    # h_final_1 = Dense(args.embedding_size, activation='softmax')(h_lstm_out)
    h_final_1 = LSTM(args.embedding_size, return_sequences=True,
                     batch_input_shape=(args.batch_size, args.episode_size, args.embedding_size))(h_lstm_out)
    final = Dense(args.final_size, activation='softmax', name='final')(h_final_1)

    # h_tone_1 = Dense(args.embedding_size, activation='sigmoid')(h_lstm_out)
    h_tone_1 = LSTM(args.embedding_size, return_sequences=True,
                    batch_input_shape=(args.batch_size, args.episode_size, args.embedding_size))(h_lstm_out)
    tone = Dense(args.tone_size, activation='softmax', name='tone')(h_tone_1)

    # h_pinyin_1 = concatenate([h_initial_1, h_final_1, h_tone_1])
    # h_pinyin_2 = Dense(args.embedding_size, activation='sigmoid')(h_pinyin_1)
    h_pinyin_1 = concatenate([h_initial_1, h_final_1, h_tone_1])
    h_pinyin_2 = LSTM(args.embedding_size, return_sequences=True,
                      batch_input_shape=(args.batch_size, args.episode_size, args.embedding_size))(h_pinyin_1)
    pinyin = Dense(args.pinyin_size, activation='softmax', name='pinyin')(h_pinyin_2)

    h_out = concatenate([h_lstm_out, h_pinyin_2])
    h_out_1 = Dense(args.embedding_size, activation='sigmoid')(h_out)
    h_out_1 = LSTM(args.embedding_size, return_sequences=True,
                   batch_input_shape=(args.batch_size, args.episode_size, args.embedding_size))(h_out)
    character = Dense(args.character_size, activation='softmax', name='character')(h_out_1)

    model = Model(inputs=input,
                  outputs=[character, pinyin, initial, final, tone])

    # initialize the optimizer
    ADAM_ = Adam(lr=args.lr)

    # compile the model
    model.compile(loss={'character': 'categorical_crossentropy',
                        'pinyin': 'categorical_crossentropy',
                        'initial': 'categorical_crossentropy',
                        'final': 'categorical_crossentropy',
                        'tone': 'categorical_crossentropy'},
                  loss_weights={'character': 0.2,
                                'pinyin': 0.2,
                                'initial': 0.2,
                                'final': 0.2,
                                'tone': 0.2},
                  optimizer=ADAM_,
                  metrics=['categorical_accuracy'])

    # save_sample_image the picture of the model
    print(time.strftime('%Y-%m-%d %H:%M:%S') +
          ' Save picture of model architecture')
    plot_model(model, show_shapes=True,
               to_file=args.model_picture)

    # show the information of the model
    print(time.strftime('%Y-%m-%d %H:%M:%S') + " Model summary")
    print(model.summary())

    return model


def get_2(args):
    # get figuration

    # Headline input: meant to receive sequences of 21 integers, between 1 and args.episode_size.
    # Note that we can name any layer by passing it a "name" argument.
    input = Input(shape=(args.episode_size,), dtype='int32', name='input')

    # This embedding layer will encode the input sequence
    # into a sequence of dense 512-dimensional vectors.
    x = Embedding(output_dim=args.embedding_size,
                  input_dim=args.vocabulary_size,
                  input_length=args.episode_size, )(input)

    # A LSTM will transform the vector sequence into a single vector,
    # containing information about the entire sequence
    h_lstm_out = LSTM(args.hidden_size, return_sequences=True,
                      batch_input_shape=(args.batch_size, args.episode_size, args.hidden_size))(x)

    # h_initial_1 = Dense(args.embedding_size, activation='softmax')(h_lstm_out)
    h_initial_1 = LSTM(args.embedding_size, return_sequences=True,
                       batch_input_shape=(args.batch_size, args.episode_size, args.embedding_size))(h_lstm_out)
    initial = Dense(args.initial_size, activation='softmax', name='initial')(h_initial_1)

    # h_final_1 = Dense(args.embedding_size, activation='softmax')(h_lstm_out)
    h_final_1 = LSTM(args.embedding_size, return_sequences=True,
                     batch_input_shape=(args.batch_size, args.episode_size, args.embedding_size))(h_lstm_out)
    final = Dense(args.final_size, activation='softmax', name='final')(h_final_1)

    # h_tone_1 = Dense(args.embedding_size, activation='sigmoid')(h_lstm_out)
    h_tone_1 = LSTM(args.embedding_size, return_sequences=True,
                    batch_input_shape=(args.batch_size, args.episode_size, args.embedding_size))(h_lstm_out)
    tone = Dense(args.tone_size, activation='softmax', name='tone')(h_tone_1)

    # h_pinyin_1 = concatenate([h_initial_1, h_final_1, h_tone_1])
    # h_pinyin_2 = Dense(args.embedding_size, activation='sigmoid')(h_pinyin_1)
    h_pinyin_1 = concatenate([h_initial_1, h_final_1, h_tone_1])
    h_pinyin_2 = LSTM(args.embedding_size, return_sequences=True,
                      batch_input_shape=(args.batch_size, args.episode_size, args.embedding_size))(h_pinyin_1)
    pinyin = Dense(args.pinyin_size, activation='softmax', name='pinyin')(h_pinyin_2)

    h_out = concatenate([h_lstm_out, h_pinyin_2])
    h_out_1 = Dense(args.embedding_size, activation='sigmoid')(h_out)
    h_out_1 = LSTM(args.embedding_size, return_sequences=True,
                   batch_input_shape=(args.batch_size, args.episode_size, args.embedding_size))(h_out)
    character = Dense(args.character_size, activation='softmax', name='character')(h_out_1)

    model = Model(inputs=input,
                  outputs=[character, pinyin, initial, final, tone])

    # initialize the optimizer
    ADAM_ = Adam(lr=args.lr)

    # compile the model
    model.compile(loss={'character': 'categorical_crossentropy',
                        'pinyin': 'categorical_crossentropy',
                        'initial': 'categorical_crossentropy',
                        'final': 'categorical_crossentropy',
                        'tone': 'categorical_crossentropy'},
                  loss_weights={'character': 0.2,
                                'pinyin': 0.2,
                                'initial': 0.2,
                                'final': 0.2,
                                'tone': 0.2},
                  optimizer=ADAM_,
                  metrics=['categorical_accuracy'])

    # save_sample_image the picture of the model
    print(time.strftime('%Y-%m-%d %H:%M:%S') +
          ' Save picture of model architecture')
    plot_model(model, show_shapes=True,
               to_file=args.model_picture)

    # show the information of the model
    print(time.strftime('%Y-%m-%d %H:%M:%S') + " Model summary")
    print(model.summary())

    return model
