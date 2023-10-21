import pandas as pd
from pandas import read_csv
from scipy.ndimage.filters import gaussian_filter1d
import matplotlib.pyplot as plt  # this is used for the plot the graph
import numpy as np

adct_fmnist_thesis25_5 = pd.read_csv(
    "adct_fmnist_thesis25_5.txt",
    sep=";",
    names=[
        "time",
        "evaltime",
        "precision",
        "recall",
        "f1-score",
        "loss_distributed",
        "loss_centralized",
        "accuracy_centralize",
        "accuracy_distributed",
        "-",
    ],
)

adct_fmnist_thesis25_10 = pd.read_csv(
    "adct_fmnist_thesis25_10.txt",
    sep=";",
    names=[
        "time",
        "evaltime",
        "precision",
        "recall",
        "f1-score",
        "loss_distributed",
        "loss_centralized",
        "accuracy_centralize",
        "accuracy_distributed",
        "-",
    ],
)

adct_fmnist_thesis25_15 = pd.read_csv(
    "adct_fmnist_thesis25_15.txt",
    sep=";",
    names=[
        "time",
        "evaltime",
        "precision",
        "recall",
        "f1-score",
        "loss_distributed",
        "loss_centralized",
        "accuracy_centralize",
        "accuracy_distributed",
        "-",
    ],
)

adct_fmnist_thesis50_5 = pd.read_csv(
    "adct_fmnist_thesis50_5.txt",
    sep=";",
    names=[
        "time",
        "evaltime",
        "precision",
        "recall",
        "f1-score",
        "loss_distributed",
        "loss_centralized",
        "accuracy_centralize",
        "accuracy_distributed",
        "-",
    ],
)

adct_fmnist_thesis50_10 = pd.read_csv(
    "adct_fmnist_thesis50_10.txt",
    sep=";",
    names=[
        "time",
        "evaltime",
        "precision",
        "recall",
        "f1-score",
        "loss_distributed",
        "loss_centralized",
        "accuracy_centralize",
        "accuracy_distributed",
        "-",
    ],
)

adct_fmnist_thesis50_15 = pd.read_csv(
    "adct_fmnist_thesis50_15.txt",
    sep=";",
    names=[
        "time",
        "evaltime",
        "precision",
        "recall",
        "f1-score",
        "loss_distributed",
        "loss_centralized",
        "accuracy_centralize",
        "accuracy_distributed",
        "-",
    ],
)

adct_fmnist_thesis75_5 = pd.read_csv(
    "adct_fmnist_thesis75_5.txt",
    sep=";",
    names=[
        "time",
        "evaltime",
        "precision",
        "recall",
        "f1-score",
        "loss_distributed",
        "loss_centralized",
        "accuracy_centralize",
        "accuracy_distributed",
        "-",
    ],
)

adct_fmnist_thesis75_10 = pd.read_csv(
    "adct_fmnist_thesis75_10.txt",
    sep=";",
    names=[
        "time",
        "evaltime",
        "precision",
        "recall",
        "f1-score",
        "loss_distributed",
        "loss_centralized",
        "accuracy_centralize",
        "accuracy_distributed",
        "-",
    ],
)

adct_fmnist_thesis75_15 = pd.read_csv(
    "adct_fmnist_thesis75_15.txt",
    sep=";",
    names=[
        "time",
        "evaltime",
        "precision",
        "recall",
        "f1-score",
        "loss_distributed",
        "loss_centralized",
        "accuracy_centralize",
        "accuracy_distributed",
        "-",
    ],
)

plt.plot(
    gaussian_filter1d(adct_fmnist_thesis25_5["accuracy_distributed"] * 100, sigma=0.75),
    ls="solid",
    linewidth=2,
    marker="p",
    markersize=5,
)
plt.plot(
    gaussian_filter1d(adct_fmnist_thesis25_10["accuracy_distributed"] * 100, sigma=0.75),
    ls="solid",
    linewidth=2,
    marker="d",
)
plt.plot(
    gaussian_filter1d(adct_fmnist_thesis25_15["accuracy_distributed"] * 100, sigma=0.75),
    ls="solid",
    linewidth=2,
    marker="d",
)
plt.plot(
    gaussian_filter1d(adct_fmnist_thesis50_5["accuracy_distributed"] * 100, sigma=0.75),
    ls="solid",
    linewidth=2,
    marker="p",
    markersize=5,
)

plt.plot(
    gaussian_filter1d(adct_fmnist_thesis50_10["accuracy_distributed"] * 100, sigma=0.75),
    ls="solid",
    linewidth=2,
    marker="d",
)
plt.plot(
    gaussian_filter1d(adct_fmnist_thesis50_15["accuracy_distributed"] * 100, sigma=0.75),
    ls="solid",
    linewidth=2,
    marker="d",
)

plt.plot(
    gaussian_filter1d(adct_fmnist_thesis75_5["accuracy_distributed"] * 100, sigma=0.75),
    ls="solid",
    linewidth=2,
    marker="p",
    markersize=5,
)
plt.plot(
    gaussian_filter1d(adct_fmnist_thesis75_10["accuracy_distributed"] * 100, sigma=0.75),
    ls="solid",
    linewidth=2,
    marker="d",
)
plt.plot(
    gaussian_filter1d(adct_fmnist_thesis75_15["accuracy_distributed"] * 100, sigma=0.75),
    ls="solid",
    linewidth=2,
    marker="d",
)

plt.xticks(np.arange(0, 51, 5))
plt.rc("grid", linestyle="dotted", color="grey")
plt.grid(True)

plt.xlabel("Communication Rounds")
plt.ylabel("Accuracy (%)")
plt.legend(["25_5","25_10","25_15", "50_5","50_10","50_15", "75_5","75_10","75_15"], loc="lower right")

plt.show()
# plt.savefig('MNIST_Acc.png',bbox_inches='tight',  dpi=700)	#save image
