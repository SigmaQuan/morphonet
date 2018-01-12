# -*- coding: utf-8 -*-
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import time
import sys
import os

current_py_file = os.path.abspath(__file__)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(current_py_file))))

from dashboard import Dashboard
from morphonets.datasets import zhwiki_corpus
import model
import config


def run():
    # set initial time
    start_time = time.time()

    # dump hyper parameters settings
    print(time.strftime('%Y-%m-%d %H:%M:%S') + ' Hyper-parameters setting')
    # get configuration
    args = config.get()

    # check for save_sample_image model
    save_best_model = ModelCheckpoint(
        filepath=args.model_file_train,
        verbose=1, save_best_only=True)  # , save_best_only=True

    if os.path.isfile(args.model_file_train):
        print(time.strftime('%Y-%m-%d %H:%M:%S') + ' Load model from file...')
        skip_gram_model = load_model(args.model_file_train)
    else:
        # get embedding model
        print(time.strftime('%Y-%m-%d %H:%M:%S') + ' Build model...')
        skip_gram_model = model.get(args)

    # dashboard
    watch_board = Dashboard(folder=config.FOLDER,
                            dump_file="dashboard.dump",
                            statistic_file="statistic.txt",
                            model=skip_gram_model,
                            show_board=True)

    # begin training
    print(time.strftime('%Y-%m-%d %H:%M:%S') + " Begin training..")
    # Train the model each generation and show predictions against the
    # validation dataset
    for iteration in range(1, args.iterations):
        print(time.strftime('%Y-%m-%d %H:%M:%S') + ' Iteration %d ' % iteration)
        skip_gram_model.fit_generator(
            zhwiki_corpus.skip_gram_generator(batch_size=args.batch_size,
                                              context_window_size=5,
                                              negative_samples=10),
            steps_per_epoch=args.steps_per_epoch,
            epochs=args.epochs,
            callbacks=[save_best_model, watch_board],
            # callbacks=[save_best_model],
            validation_data=zhwiki_corpus.skip_gram_generator(
                batch_size=args.batch_size,
                context_window_size=5,
                negative_samples=10),
            validation_steps=100,
            verbose=0)

    # close_board windows
    watch_board.close_board()
    print("task took %.3fs" % (float(time.time()) - start_time))


# log_every = False
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
