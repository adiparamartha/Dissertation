import pandas as pd
from pandas import read_csv
from scipy.ndimage.filters import gaussian_filter1d
import matplotlib.pyplot as plt  # this is used for the plot the graph
import numpy as np


ecsm_cifar1018 = pd.read_csv(
    "ECSMCIFAR10_1_8.txt",
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

ecsm_cifar1027 = pd.read_csv(
    "ECSMCIFAR10_2_7.txt",
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

ecsm_cifar1036 = pd.read_csv(
    "ECSMCIFAR10_3_6.txt",
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

ecsm_cifar1045 = pd.read_csv(
    "ECSMCIFAR10_4_5.txt",
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

ecsm_cifar1054 = pd.read_csv(
    "ECSMCIFAR10_5_4.txt",
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

ecsm_cifar1063 = pd.read_csv(
    "ECSMCIFAR10_6_3.txt",
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

ecsm_cifar1072 = pd.read_csv(
    "ECSMCIFAR10_7_2.txt",
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

ecsm_cifar1081 = pd.read_csv(
    "ECSMCIFAR10_8_1.txt",
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
    gaussian_filter1d(ecsm_cifar1018["accuracy_distributed"] * 100, sigma=0.75),
    ls="solid",
    linewidth=2,
    marker="p",
    markersize=5,
)
plt.plot(
    gaussian_filter1d(ecsm_cifar1027["accuracy_distributed"] * 100, sigma=0.75),
    ls="solid",
    linewidth=2,
    marker="d",
)
plt.plot(
    gaussian_filter1d(ecsm_cifar1036["accuracy_distributed"] * 100, sigma=0.75),
    ls="solid",
    linewidth=2,
    marker="d",
)
plt.plot(
    gaussian_filter1d(ecsm_cifar1045["accuracy_distributed"] * 100, sigma=0.75),
    ls="solid",
    linewidth=2,
    marker="p",
    markersize=5,
)

plt.plot(
    gaussian_filter1d(ecsm_cifar1054["accuracy_distributed"] * 100, sigma=0.75),
    ls="solid",
    linewidth=2,
    marker="d",
)
plt.plot(
    gaussian_filter1d(ecsm_cifar1063["accuracy_distributed"] * 100, sigma=0.75),
    ls="solid",
    linewidth=2,
    marker="d",
)

plt.plot(
    gaussian_filter1d(ecsm_cifar1072["accuracy_distributed"] * 100, sigma=0.75),
    ls="solid",
    linewidth=2,
    marker="p",
    markersize=5,
)
plt.plot(
    gaussian_filter1d(ecsm_cifar1081["accuracy_distributed"] * 100, sigma=0.75),
    ls="solid",
    linewidth=2,
    marker="d",
)
print("Accuracy, Precision, Recall")
print(round(ecsm_cifar1018["accuracy_distributed"][74] * 100,2), round(ecsm_cifar1018["precision"][74] * 100,2), round(ecsm_cifar1018["recall"][74] * 100,2) )
print(round(ecsm_cifar1027["accuracy_distributed"][74] * 100,2), round(ecsm_cifar1027["precision"][74] * 100,2), round(ecsm_cifar1027["recall"][74] * 100,2) )
print(round(ecsm_cifar1036["accuracy_distributed"][74] * 100,2), round(ecsm_cifar1036["precision"][74] * 100,2), round(ecsm_cifar1036["recall"][74] * 100,2) )
print(round(ecsm_cifar1045["accuracy_distributed"][74] * 100,2), round(ecsm_cifar1045["precision"][74] * 100,2), round(ecsm_cifar1045["recall"][74] * 100,2) )
print(round(ecsm_cifar1054["accuracy_distributed"][74] * 100,2), round(ecsm_cifar1054["precision"][74] * 100,2), round(ecsm_cifar1054["recall"][74] * 100,2) )
print(round(ecsm_cifar1063["accuracy_distributed"][74] * 100,2), round(ecsm_cifar1063["precision"][74] * 100,2), round(ecsm_cifar1063["recall"][74] * 100,2) )
print(round(ecsm_cifar1072["accuracy_distributed"][74] * 100,2), round(ecsm_cifar1072["precision"][74] * 100,2), round(ecsm_cifar1072["recall"][74] * 100,2) )
print(round(ecsm_cifar1081["accuracy_distributed"][74] * 100,2), round(ecsm_cifar1081["precision"][74] * 100,2), round(ecsm_cifar1081["recall"][74] * 100,2) )


plt.xticks(np.arange(0, 76, 5))
plt.rc("grid", linestyle="dotted", color="grey")
plt.grid(True)

plt.xlabel("Communication Rounds")
plt.ylabel("Accuracy (%)")
plt.legend(["1_8","2_7","3_6", "4_5","5_4","6_3", "7_2","8_1"], loc="lower right")

plt.show()
# plt.savefig('CIFAR10_Acc.png',bbox_inches='tight',  dpi=700)	#save image
