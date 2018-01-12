# -*- coding: utf-8 -*-

# from keras.layers import Input, Embedding, LSTM, Dense, concatenate
# from keras.models import Model
# from keras.models import Sequential
# from keras.layers import Dense, recurrent
# from keras.optimizers import Adam
# from keras.utils.vis_utils import plot_model
# from keras.metrics import categorical_accuracy
# from keras.losses import categorical_crossentropy
# from knowledge import cc_phonology
# from keras.optimizers import Adam
# from keras.utils.vis_utils import plot_model

from morphonets.datasets import zhwiki_corpus
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import time
import sys
import os
import model
import config
from morphonets.morphonet1.dashboard import Dashboard


def run():
    start_time = time.time()

    # dump hyper parameters settings
    print(time.strftime('%Y-%m-%d %H:%M:%S') + ' Hyper-parameters setting')
    # get configuration
    args = config.get()

    # check for save_sample_image model
    save_best_model = ModelCheckpoint(
        filepath=args.model_file_train,
        verbose=1)  # , save_best_only=True

    if os.path.isfile(args.model_file_train):
        print(time.strftime('%Y-%m-%d %H:%M:%S') + ' Load model from file...')
        embeding_model = load_model(args.model_file_train)
    else:
        # get embedding model
        print(time.strftime('%Y-%m-%d %H:%M:%S') + ' Build model...')
        embeding_model = model.get(args)

    # dashboard
    watch_board = Dashboard(folder=config.FOLDER,
                            statistic_file="statistic.txt",
                            model=embeding_model,
                            show_board=True)

    # begin training
    print(time.strftime('%Y-%m-%d %H:%M:%S') + " Begin training..")
    # Train the model each generation and show predictions against the
    # validation dataset
    for iteration in range(1, args.iterations):
        print(time.strftime('%Y-%m-%d %H:%M:%S') + ' Iteration %d ' % iteration)
        embeding_model.fit_generator(
            zhwiki_corpus.generator(batch_size=args.batch_size,
                                    max_token_length=args.max_token_length),
            steps_per_epoch=args.steps_per_epoch,
            epochs=args.epochs,
            callbacks=[save_best_model, watch_board],
            # callbacks=[save_best_model],
            validation_data=zhwiki_corpus.generator(
                batch_size=args.batch_size,
                max_token_length=args.max_token_length),
            validation_steps=100,
            verbose=1)

    # close_board windows
    # dashboard.close_board()
    print("task took %.3fs" % (float(time.time()) - start_time))


log_every = True
# create log file
if log_every:
    sys_stdout = sys.stdout
    log_file = '%s/training_embedding.log' % config.FOLDER
    sys.stdout = open(log_file, 'a')

# training
run()

# create log file
if log_every:
    sys.stdout.close_board()
    sys.stdout = sys_stdout
