# -*- coding: utf-8 -*-
"""
Monitoring the progress of training Chinese word embeddings with skip-gram.
"""
# from matplotlib.backends.backend_pdf import PdfPages
# from visualization import make_tick_labels_invisible
# from datasets import associative_recall
# import numpy as np
# import time
from keras.callbacks import Callback
import os
import pickle

import sys
current_py_file = os.path.abspath(__file__)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(current_py_file))))
from configs.globals import PROJECT_FOLDER
sys.path.append(PROJECT_FOLDER)
import morphonets.datasets.joint_evaluation as test


def dump_dashboard(dump_file, dashboard):
    """Save the content of a dashboard to a file.

    # Arguments
        dump_file: the file path.
        dashboard: a dashboard.
    """
    with open(dump_file, 'w') as f:
        pickle.dump(dashboard.epochs, f, protocol=2)
        pickle.dump(dashboard.loss_train, f, protocol=2)
        pickle.dump(dashboard.loss_valid, f, protocol=2)
        pickle.dump(dashboard.acc_wsc_rho_1, f, protocol=2)
        pickle.dump(dashboard.acc_wsc_pval_1, f, protocol=2)
        pickle.dump(dashboard.acc_wsc_rho_2, f, protocol=2)
        pickle.dump(dashboard.acc_wsc_pval_2, f, protocol=2)
        pickle.dump(dashboard.acc_war_total, f, protocol=2)
        pickle.dump(dashboard.acc_war_capital, f, protocol=2)
        pickle.dump(dashboard.acc_war_state, f, protocol=2)
        pickle.dump(dashboard.acc_war_family, f, protocol=2)


def load_dashboard(dump_file, dashboard):
    """Load the content of dashboard from a file.

    # Arguments
        dump_file: the file path.

    # Returns
        dashboard: a dashboard.
    """
    with open(dump_file, 'r') as f:
        dashboard.epochs = pickle.load(f)
        dashboard.loss_train = pickle.load(f)
        dashboard.loss_valid = pickle.load(f)
        dashboard.acc_wsc_rho_1 = pickle.load(f)
        dashboard.acc_wsc_pval_1 = pickle.load(f)
        dashboard.acc_wsc_rho_2 = pickle.load(f)
        dashboard.acc_wsc_pval_2 = pickle.load(f)
        dashboard.acc_war_total = pickle.load(f)
        dashboard.acc_war_capital = pickle.load(f)
        dashboard.acc_war_state = pickle.load(f)
        dashboard.acc_war_family = pickle.load(f)

    return dashboard


class Dashboard(Callback):
    """Create a dashboard for monitoring loss, accuracy, etc..., during the
    process of learning embedding model.

    # Arguments
        folder: the folder for saving sample picture file and statistic txt
            file.
        dump_file: the file for saving the content of the dashboard.
        statistic_file: the txt file for saving loss, accuracy on training,
            testing and validation.
        model: a machine learning model.
        show_board: whether show the board during the training.
        figure_size_loss: the size of figure for watching skip-gram loss .
        figure_size_wsc: the size of figure for watching the accuracy of word
            similarity computation.
        figure_size_wsc: the size of figure for watching the accuracy of word
            analogy reasoning.
    """
    def __init__(self, folder, dump_file, statistic_file, model, show_board=True,
                 figure_size_loss=(6.5, 5), figure_size_wsc=(6.5, 5),
                 figure_size_war=(6.5, 5)):
        super(Dashboard, self).__init__()

        self.folder = folder
        if not os.path.isdir(self.folder):
            os.makedirs(self.folder)
            print("Create folder: %s" % self.folder)
        self.statistic_file = os.path.join(self.folder, statistic_file)
        self.dump_file = os.path.join(self.folder, dump_file)

        self.model = model
        self.show_board = show_board

        # set figures
        self.fig_size_loss = figure_size_loss
        self.fig_size_wsc = figure_size_wsc
        self.fig_size_war = figure_size_war
        self.fig_loss, self.ax_loss = None, None
        self.fig_wsc, self.ax_wsc = None, None
        self.fig_war, self.ax_war = None, None

        # the losses of training and testing loss of skip-gram
        self.loss_train = []
        self.loss_valid = []

        # indexes of word similarity computation
        self.acc_wsc_rho_1 = []
        self.acc_wsc_pval_1 = []
        self.acc_wsc_rho_2 = []
        self.acc_wsc_pval_2 = []

        # indexes of word analogy reasoning
        self.acc_war_total = []
        self.acc_war_capital = []
        self.acc_war_state = []
        self.acc_war_family = []

        # number of epochs
        self.epochs = 0

        # whether show dashboard
        if self.show_board:
            import matplotlib.pyplot as plt
            self.plt = plt
            self.plt.style.use("ggplot")
            self.init_figures()

    def init_figures(self):
        self.fig_loss, self.ax_loss = self.init_figure_size(self.fig_size_loss)
        self.fig_wsc, self.ax_wsc = self.init_figure_size(self.fig_size_wsc)
        self.fig_war, self.ax_war = self.init_figure_size(self.fig_size_war)

        self.plt.ion()
        self.update_board()

    def init_figure_size(self, fig_size):
        if self.show_board:
            return self.plt.subplots(figsize=fig_size)
        else:
            return None, None

    def on_train_end(self, epoch, logs=None):
        self.save_statistics()
        # dump_dashboard(self.dump_file, self)

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
        # save statistics
        self.save_statistics()

    def get_monitors(self, logs):
        # get training and validation losses
        current_train_loss = logs.get('loss')
        current_valid_loss = logs.get('val_loss')
        self.loss_train.append(current_train_loss)
        self.loss_valid.append(current_valid_loss)

        vectors = self.model.get_weights()[0]
        acc_wsc_rho_1, acc_wsc_pval_1, \
        acc_wsc_rho_2, acc_wsc_pval_2, \
        acc_war_total, acc_war_capital, \
        acc_war_state, acc_war_family = test.general_measure(vectors)

        self.acc_wsc_rho_1.append(acc_wsc_rho_1)
        self.acc_wsc_pval_1.append(acc_wsc_pval_1)
        self.acc_wsc_rho_2.append(acc_wsc_rho_2)
        self.acc_wsc_pval_2.append(acc_wsc_pval_2)
        self.acc_war_total.append(acc_war_total)
        self.acc_war_capital.append(acc_war_capital)
        self.acc_war_state.append(acc_war_state)
        self.acc_war_family.append(acc_war_family)

    def update_board(self):
        # update_board watch characteristics
        self.update_indicators_4_dashboard()
        # show figure
        self.plt.show()
        # plt.draw()
        self.plt.pause(0.25)

    def update_indicators_4_dashboard(self):
        self.ax_loss.cla()
        self.ax_loss.title.set_text("Skip-Gram")
        self.ax_loss.set_xlabel("Epoch #")
        self.ax_loss.set_ylabel("Loss")
        # self.ax_loss.grid()
        self.ax_loss.plot(self.loss_train, label='train')
        self.ax_loss.plot(self.loss_valid, label='valid')
        self.ax_loss.legend(loc='best')

        self.ax_wsc.cla()
        self.ax_wsc.title.set_text("Word Similarity Computation")
        self.ax_wsc.set_xlabel("Epoch #")
        self.ax_wsc.set_ylabel("Acc")
        # self.ax_wsc.grid()
        self.ax_wsc.plot(self.acc_wsc_rho_1, label='word sim 240')
        # self.ax_wsc.plot(self.acc_wsc_rho_1, label='rho_1')
        # self.ax_wsc.plot(self.acc_wsc_pval_1, label='pval_1')
        self.ax_wsc.plot(self.acc_wsc_rho_2, label='word sim 296')
        # self.ax_wsc.plot(self.acc_wsc_rho_2, label='rho_2')
        # self.ax_wsc.plot(self.acc_wsc_pval_2, label='pval_2')
        self.ax_wsc.legend(loc='best')

        self.ax_war.cla()
        self.ax_war.title.set_text("Word Analogy Reasoning")
        self.ax_war.set_xlabel("Epoch #")
        self.ax_war.set_ylabel("Acc")
        # self.ax_war.grid()
        self.ax_war.plot(self.acc_war_total, label='total')
        self.ax_war.plot(self.acc_war_capital, label='capital')
        self.ax_war.plot(self.acc_war_state, label='state')
        self.ax_war.plot(self.acc_war_family, label='family')
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
            f.write("epochs:" + "\n")
            f.write(self.epochs.__str__() + "\n\n")

            f.write("train loss: " + "\n")
            f.write(self.loss_train.__str__() + "\n\n")
            f.write("valid loss: " + "\n")
            f.write(self.loss_valid.__str__() + "\n\n")

            f.write("wsc_rho_1: " + "\n")
            f.write(self.acc_wsc_rho_1.__str__() + "\n\n")
            f.write("wsc_pval_1: " + "\n")
            f.write(self.acc_wsc_pval_1.__str__() + "\n\n")

            f.write("wsc_rho_2: " + "\n")
            f.write(self.acc_wsc_rho_2.__str__() + "\n\n")
            f.write("wsc_pval_2 loss: " + "\n")
            f.write(self.acc_wsc_pval_2.__str__() + "\n\n")

            f.write("war_total: " + "\n")
            f.write(self.acc_war_total.__str__() + "\n\n")
            f.write("war_capital: " + "\n")
            f.write(self.acc_war_capital.__str__() + "\n\n")
            f.write("war_state: " + "\n")
            f.write(self.acc_war_state.__str__() + "\n\n")
            f.write("war_family: " + "\n")
            f.write(self.acc_war_family.__str__() + "\n\n")

    def close_board(self):
        # close_board plot GUI
        self.plt.close()
