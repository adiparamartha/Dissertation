import pandas as pd
from pandas import read_csv
from scipy.ndimage.filters import gaussian_filter1d
import matplotlib.pyplot as plt  # this is used for the plot the graph
import numpy as np


ecsm_mnist18 = pd.read_csv(
    "ECSMMNIST_1_8.txt",
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

ecsm_mnist27 = pd.read_csv(
    "ECSMMNIST_2_7.txt",
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

ecsm_mnist36 = pd.read_csv(
    "ECSMMNIST_3_6.txt",
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

ecsm_mnist45 = pd.read_csv(
    "ECSMMNIST_4_5.txt",
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

ecsm_mnist54 = pd.read_csv(
    "ECSMMNIST_5_4.txt",
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

ecsm_mnist63 = pd.read_csv(
    "ECSMMNIST_6_3.txt",
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

ecsm_mnist72 = pd.read_csv(
    "ECSMMNIST_7_2.txt",
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

ecsm_mnist81 = pd.read_csv(
    "ECSMMNIST_8_1.txt",
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
    gaussian_filter1d(ecsm_mnist18["accuracy_distributed"] * 100, sigma=0.75),
    ls="solid",
    linewidth=2,
    marker="p",
    markersize=5,
)
plt.plot(
    gaussian_filter1d(ecsm_mnist27["accuracy_distributed"] * 100, sigma=0.75),
    ls="solid",
    linewidth=2,
    marker="d",
)
plt.plot(
    gaussian_filter1d(ecsm_mnist36["accuracy_distributed"] * 100, sigma=0.75),
    ls="solid",
    linewidth=2,
    marker="d",
)
plt.plot(
    gaussian_filter1d(ecsm_mnist45["accuracy_distributed"] * 100, sigma=0.75),
    ls="solid",
    linewidth=2,
    marker="p",
    markersize=5,
)

plt.plot(
    gaussian_filter1d(ecsm_mnist54["accuracy_distributed"] * 100, sigma=0.75),
    ls="solid",
    linewidth=2,
    marker="d",
)
plt.plot(
    gaussian_filter1d(ecsm_mnist63["accuracy_distributed"] * 100, sigma=0.75),
    ls="solid",
    linewidth=2,
    marker="d",
)

plt.plot(
    gaussian_filter1d(ecsm_mnist72["accuracy_distributed"] * 100, sigma=0.75),
    ls="solid",
    linewidth=2,
    marker="p",
    markersize=5,
)
plt.plot(
    gaussian_filter1d(ecsm_mnist81["accuracy_distributed"] * 100, sigma=0.75),
    ls="solid",
    linewidth=2,
    marker="d",
)

print("Accuracy, Precision, Recall")
print(round(ecsm_mnist18["accuracy_distributed"][49] * 100,2), round(ecsm_mnist18["precision"][49] * 100,2), round(ecsm_mnist18["recall"][49] * 100,2) )
print(round(ecsm_mnist27["accuracy_distributed"][49] * 100,2), round(ecsm_mnist27["precision"][49] * 100,2), round(ecsm_mnist27["recall"][49] * 100,2) )
print(round(ecsm_mnist36["accuracy_distributed"][49] * 100,2), round(ecsm_mnist36["precision"][49] * 100,2), round(ecsm_mnist36["recall"][49] * 100,2) )
print(round(ecsm_mnist45["accuracy_distributed"][49] * 100,2), round(ecsm_mnist45["precision"][49] * 100,2), round(ecsm_mnist45["recall"][49] * 100,2) )
print(round(ecsm_mnist54["accuracy_distributed"][49] * 100,2), round(ecsm_mnist54["precision"][49] * 100,2), round(ecsm_mnist54["recall"][49] * 100,2) )
print(round(ecsm_mnist63["accuracy_distributed"][49] * 100,2), round(ecsm_mnist63["precision"][49] * 100,2), round(ecsm_mnist63["recall"][49] * 100,2) )
print(round(ecsm_mnist72["accuracy_distributed"][49] * 100,2), round(ecsm_mnist72["precision"][49] * 100,2), round(ecsm_mnist72["recall"][49] * 100,2) )
print(round(ecsm_mnist81["accuracy_distributed"][49] * 100,2), round(ecsm_mnist81["precision"][49] * 100,2), round(ecsm_mnist81["recall"][49] * 100,2) )


plt.xticks(np.arange(0, 51, 5))
plt.rc("grid", linestyle="dotted", color="grey")
plt.grid(True)

plt.xlabel("Communication Rounds")
plt.ylabel("Accuracy (%)")
plt.legend(["1_8","2_7","3_6", "4_5","5_4","6_3", "7_2","8_1"], loc="lower right")

plt.show()
# plt.savefig('MNIST_Acc.png',bbox_inches='tight',  dpi=700)	#save image
