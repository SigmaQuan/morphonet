# -*- coding: utf-8 -*-
"""
Monitoring the progress of training Chinese word embeddings with morphonet-1.
"""
# from matplotlib.backends.backend_pdf import PdfPages
# from visualization import make_tick_labels_invisible
# from datasets import associative_recall
import numpy as np
import time
from keras.callbacks import Callback
import os
import sys

current_py_file = os.path.abspath(__file__)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(current_py_file))))

# from elwm.datasets.ptb import to_text
import morphonets.datasets.joint_evaluation as test
from morphonets.datasets import zhwiki_corpus
from morphonets.knowledge import cc_phonology

# get dictionaries
id_pinyins = cc_phonology.get_id_pinyin_dic()
id_initials, id_finals, id_tones = cc_phonology.get_id_other_dic()
_, id_word_dic = zhwiki_corpus.get_word_id_dictionaries()
dictionaries = (id_word_dic, id_pinyins, id_initials, id_finals, id_tones)


class Dashboard(Callback):
    """Create a dashboard for monitoring loss, accuracy, etc..., during the
    process of learning embedding model.

    # Arguments
        folder: the folder for saving sample picture file and statistic txt
            file.
        statistic_file: the txt file for saving loss, accuracy on training,
            testing and validation.
        model: a ANN model.
        watch_sample: whether watch prediction of samples during the training.
        show_board: whether watch loss curve during the training.
        watch_metric: whether watch metric curve during the training.
        metric_name: the name of metric.
        test_data: the test data.
        steps_per_epoch_test: the steps per epoch for testing,
            i.e. # test_sample / # batch_size.
        visualized_sample_size: the number of samples which be displayed during
            each epoch.
        sample_figure_size: the size of sample figure.
        loss_figure_size: the size of loss figure.
        metric_figure_size: the size of metric figure.
        corpus_name: the name of corpus.
        vocab: the vocabulary of the corpus.
        predicted_sample_file: a text file for saving predicted samples.
    """
    def __init__(self, folder, statistic_file, model,
                 show_board=True,
                 loss_figure_size=(6.5, 5),
                 wsc_figure_size=(6.5, 5),
                 war_figure_size=(6.5, 5)):

        super(Dashboard, self).__init__()

        self.folder = folder
        if not os.path.isdir(self.folder):
            os.makedirs(self.folder)
            print("Create folder: %s" % self.folder)
        self.statistic_file = os.path.join(self.folder, statistic_file)
        self.model = model

        self.show_board = show_board

        # set figures
        self.fig_size_loss = loss_figure_size
        self.fig_size_wsc = wsc_figure_size
        self.fig_size_war = war_figure_size
        self.fig_loss, self.ax_loss = None, None
        self.fig_wsc, self.ax_wsc = None, None
        self.fig_war, self.ax_war = None, None

        self.loss_train = []
        self.loss_valid = []
        self.loss_wsc_rho_1 = []
        self.loss_wsc_pval_1 = []
        self.loss_wsc_rho_2 = []
        self.loss_wsc_pval_2 = []
        self.loss_war_total = []
        self.loss_war_capital = []
        self.loss_war_state = []
        self.loss_war_family = []

        self.epochs = 0

        if self.show_board:
            import matplotlib.pyplot as plt
            self.plt = plt
            self.plt.style.use("ggplot")
            self.init_figures()

    def init_figures(self):
        self.fig_loss, self.ax_loss = self.init_fig_loss()
        self.fig_wsc, self.ax_wsc = self.init_fig_wsc()
        self.fig_war, self.ax_war = self.init_fig_war()

        self.plt.ion()
        self.update_board()

    def init_fig_loss(self):
        if self.show_board:
            return self.plt.subplots(figsize=self.fig_size_loss)
        else:
            return None, None

    def init_fig_wsc(self):
        if self.show_board:
            return self.plt.subplots(figsize=self.fig_size_wsc)
        else:
            return None, None

    def init_fig_war(self):
        if self.show_board:
            return self.plt.subplots(figsize=self.fig_size_war)
        else:
            return None, None

    def set_model(self, model):
        self.model = model

    def on_train_end(self, epoch, logs=None):
        self.save_statistics()

    def on_epoch_end(self, epoch, logs=None):
        # set logs
        logs = logs or {}
        # update_board epochs
        self.epochs += 1
        # get statistics
        self.get_monitors(logs)
        # update_board dashboard
        if self.show_board:
            self.update_board()

    def get_monitors(self, logs):
        # get training and validation losses
        current_train_loss = logs.get('loss')
        current_validation_loss = logs.get('val_loss')
        self.loss_train.append(current_train_loss)
        self.loss_valid.append(current_validation_loss)

        vectors = self.model.get_weights()[0]
        loss_wsc_rho_1, loss_wsc_pval_1, \
        loss_wsc_rho_2, loss_wsc_pval_2, \
        loss_war_total, loss_war_capital, \
        loss_war_state, loss_war_family = test.general_measure(vectors)
        # acc_war_state, acc_war_family = test.general_measure(self.model)

        self.loss_wsc_rho_1.append(loss_wsc_rho_1)
        self.loss_wsc_pval_1.append(loss_wsc_pval_1)
        self.loss_wsc_rho_2.append(loss_wsc_rho_2)
        self.loss_wsc_pval_2.append(loss_wsc_pval_2)
        self.loss_war_total.append(loss_war_total)
        self.loss_war_capital.append(loss_war_capital)
        self.loss_war_state.append(loss_war_state)
        self.loss_war_family.append(loss_war_family)

    def update_board(self):
        # update_board watch characteristics
        self.update_loss()
        # show figure
        self.plt.show()
        # plt.draw()
        self.plt.pause(0.25)

    def update_loss(self):
        self.ax_loss.cla()
        self.ax_loss.set_xlabel("Epoch")
        self.ax_loss.set_ylabel("Loss")
        self.ax_loss.grid()
        self.ax_loss.plot(self.loss_train, label='train')
        self.ax_loss.plot(self.loss_valid, label='valid')
        self.ax_loss.legend(loc='best')

        self.ax_wsc.cla()
        self.ax_wsc.set_xlabel("Epoch")
        self.ax_wsc.set_ylabel("Acc")
        self.ax_wsc.grid()
        self.ax_wsc.plot(self.loss_wsc_rho_1, label='rho_1')
        # self.ax_wsc.plot(self.acc_wsc_pval_1, label='pval_1')
        self.ax_wsc.plot(self.loss_wsc_rho_2, label='rho_2')
        # self.ax_wsc.plot(self.acc_wsc_pval_2, label='pval_2')
        self.ax_wsc.legend(loc='best')

        self.ax_war.cla()
        self.ax_war.set_xlabel("Epoch")
        self.ax_war.set_ylabel("Acc")
        self.ax_war.grid()
        self.ax_war.plot(self.loss_war_total, label='total')
        self.ax_war.plot(self.loss_war_capital, label='capital')
        self.ax_war.plot(self.loss_war_state, label='state')
        self.ax_war.plot(self.loss_war_family, label='family')
        self.ax_war.legend(loc='best')

    def save_statistics(self):
        if self.show_board:
            file_path = self.folder + "loss.png"
            self.fig_loss.savefig(file_path, dpi=75)
            file_path_wsc = self.folder + "acc_wsc.png"
            self.fig_wsc.savefig(file_path_wsc, dpi=75)
            file_path_war = self.folder + "acc_war.png"
            self.fig_war.savefig(file_path_war, dpi=75)

        with open(self.statistic_file, 'w') as f:
            f.write("train loss: " + "\n")
            f.write(self.loss_train.__str__() + "\n\n")
            f.write("valid loss: " + "\n")
            f.write(self.loss_valid.__str__() + "\n\n")

            f.write("wsc_rho_1: " + "\n")
            f.write(self.loss_wsc_rho_1.__str__() + "\n\n")
            f.write("wsc_pval_1: " + "\n")
            f.write(self.loss_wsc_pval_1.__str__() + "\n\n")

            f.write("wsc_rho_2: " + "\n")
            f.write(self.loss_wsc_rho_2.__str__() + "\n\n")
            f.write("wsc_pval_2 loss: " + "\n")
            f.write(self.loss_wsc_pval_2.__str__() + "\n\n")

            f.write("war_total: " + "\n")
            f.write(self.loss_war_total.__str__() + "\n\n")

            f.write("war_capital: " + "\n")
            f.write(self.loss_war_capital.__str__() + "\n\n")

            f.write("war_state: " + "\n")
            f.write(self.loss_war_state.__str__() + "\n\n")

            f.write("war_family: " + "\n")
            f.write(self.loss_war_family.__str__() + "\n\n")

            f.write("epochs:" + "\n")
            f.write(self.epochs.__str__() + "\n\n")

    def close_board(self):
        # close_board plot GUI
        self.plt.close()



