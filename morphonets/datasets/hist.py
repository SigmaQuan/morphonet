from numpy.random import beta
import matplotlib.pyplot as plt


plt.style.use('bmh')


def plot_beta_hist(ax, ndarray):
    ax.hist(ndarray, histtype="stepfilled",
            bins=25, alpha=0.8, normed=True)


def hist(ndarray):
    fig, ax = plt.subplots()
    plot_beta_hist(ax, ndarray)
    ax.set_title("'bmh' style sheet")
    plt.show()
