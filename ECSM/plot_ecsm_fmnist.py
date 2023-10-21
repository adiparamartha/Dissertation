import pandas as pd
from pandas import read_csv
from scipy.ndimage.filters import gaussian_filter1d
import matplotlib.pyplot as plt  # this is used for the plot the graph
import numpy as np


ecsm_fmnist18 = pd.read_csv(
    "ECSMFMNIST_1_8.txt",
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

ecsm_fmnist27 = pd.read_csv(
    "ECSMFMNIST_2_7.txt",
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

ecsm_fmnist36 = pd.read_csv(
    "ECSMFMNIST_3_6.txt",
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

ecsm_fmnist45 = pd.read_csv(
    "ECSMFMNIST_4_5.txt",
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

ecsm_fmnist54 = pd.read_csv(
    "ECSMFMNIST_5_4.txt",
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

ecsm_fmnist63 = pd.read_csv(
    "ECSMFMNIST_6_3.txt",
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

ecsm_fmnist72 = pd.read_csv(
    "ECSMFMNIST_7_2.txt",
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

ecsm_fmnist81 = pd.read_csv(
    "ECSMFMNIST_8_1.txt",
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
    gaussian_filter1d(ecsm_fmnist18["accuracy_distributed"] * 100, sigma=0.75),
    ls="solid",
    linewidth=2,
    marker="p",
    markersize=5,
)
plt.plot(
    gaussian_filter1d(ecsm_fmnist27["accuracy_distributed"] * 100, sigma=0.75),
    ls="solid",
    linewidth=2,
    marker="d",
)
plt.plot(
    gaussian_filter1d(ecsm_fmnist36["accuracy_distributed"] * 100, sigma=0.75),
    ls="solid",
    linewidth=2,
    marker="d",
)
plt.plot(
    gaussian_filter1d(ecsm_fmnist45["accuracy_distributed"] * 100, sigma=0.75),
    ls="solid",
    linewidth=2,
    marker="p",
    markersize=5,
)

plt.plot(
    gaussian_filter1d(ecsm_fmnist54["accuracy_distributed"] * 100, sigma=0.75),
    ls="solid",
    linewidth=2,
    marker="d",
)
plt.plot(
    gaussian_filter1d(ecsm_fmnist63["accuracy_distributed"] * 100, sigma=0.75),
    ls="solid",
    linewidth=2,
    marker="d",
)

plt.plot(
    gaussian_filter1d(ecsm_fmnist72["accuracy_distributed"] * 100, sigma=0.75),
    ls="solid",
    linewidth=2,
    marker="p",
    markersize=5,
)
plt.plot(
    gaussian_filter1d(ecsm_fmnist81["accuracy_distributed"] * 100, sigma=0.75),
    ls="solid",
    linewidth=2,
    marker="d",
)

print("Accuracy, Precision, Recall")
print(round(ecsm_fmnist18["accuracy_distributed"][49] * 100,2), round(ecsm_fmnist18["precision"][49] * 100,2), round(ecsm_fmnist18["recall"][49] * 100,2) )
print(round(ecsm_fmnist27["accuracy_distributed"][49] * 100,2), round(ecsm_fmnist27["precision"][49] * 100,2), round(ecsm_fmnist27["recall"][49] * 100,2) )
print(round(ecsm_fmnist36["accuracy_distributed"][49] * 100,2), round(ecsm_fmnist36["precision"][49] * 100,2), round(ecsm_fmnist36["recall"][49] * 100,2) )
print(round(ecsm_fmnist45["accuracy_distributed"][49] * 100,2), round(ecsm_fmnist45["precision"][49] * 100,2), round(ecsm_fmnist45["recall"][49] * 100,2) )
print(round(ecsm_fmnist54["accuracy_distributed"][49] * 100,2), round(ecsm_fmnist54["precision"][49] * 100,2), round(ecsm_fmnist54["recall"][49] * 100,2) )
print(round(ecsm_fmnist63["accuracy_distributed"][49] * 100,2), round(ecsm_fmnist63["precision"][49] * 100,2), round(ecsm_fmnist63["recall"][49] * 100,2) )
print(round(ecsm_fmnist72["accuracy_distributed"][49] * 100,2), round(ecsm_fmnist72["precision"][49] * 100,2), round(ecsm_fmnist72["recall"][49] * 100,2) )
print(round(ecsm_fmnist81["accuracy_distributed"][49] * 100,2), round(ecsm_fmnist81["precision"][49] * 100,2), round(ecsm_fmnist81["recall"][49] * 100,2) )


plt.xticks(np.arange(0, 51, 5))
plt.rc("grid", linestyle="dotted", color="grey")
plt.grid(True)

plt.xlabel("Communication Rounds")
plt.ylabel("Accuracy (%)")
plt.legend(["1_8","2_7","3_6", "4_5","5_4","6_3", "7_2","8_1"], loc="lower right")

plt.show()
# plt.savefig('FMNIST_Acc.png',bbox_inches='tight',  dpi=700)	#save image
