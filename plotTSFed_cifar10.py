import pandas as pd
from pandas import read_csv
from scipy.ndimage.filters import gaussian_filter1d
import matplotlib.pyplot as plt  # this is used for the plot the graph
import numpy as np

maxline = 75
sigmaPlot = 0.75


TSFed = pd.read_csv("TSFed/tsfed_cifar10_thesis.txt", sep=";", names=["time","evaltime","precision","recall","f1-score","loss_distributed","loss_centralized","accuracy_centralize","accuracy_distributed", "-",],)
FedAvg = pd.read_csv("Baseline/baseline_CIFAR10_fedAvg.txt", sep=";", names=["time","evaltime","precision","recall","f1-score","loss_distributed","loss_centralized","accuracy_centralize","accuracy_distributed", "-",],)
FedAvgLoss = pd.read_csv("Baseline/baseline_CIFAR10_fedAvgLoss.txt", sep=";", names=["time","evaltime","precision","recall","f1-score","loss_distributed","loss_centralized","accuracy_centralize","accuracy_distributed", "-",],)
FedACS = pd.read_csv("Baseline/baseline_CIFAR10_fedAvgAcc.txt", sep=";", names=["time","evaltime","precision","recall","f1-score","loss_distributed","loss_centralized","accuracy_centralize","accuracy_distributed", "-",],)
FedOpt = pd.read_csv("Baseline/baseline_CIFAR10_fedOpt.txt", sep=";", names=["time","evaltime","precision","recall","f1-score","loss_distributed","loss_centralized","accuracy_centralize","accuracy_distributed", "-",],)
FedYogi = pd.read_csv("Baseline/baseline_CIFAR10_fedYogi.txt", sep=";", names=["time","evaltime","precision","recall","f1-score","loss_distributed","loss_centralized","accuracy_centralize","accuracy_distributed", "-",],)
FedTrim = pd.read_csv("Baseline/baseline_CIFAR10_fedTrimMedAvg.txt", sep=";", names=["time","evaltime","precision","recall","f1-score","loss_distributed","loss_centralized","accuracy_centralize","accuracy_distributed", "-",],)
FedMed = pd.read_csv("Baseline/baseline_CIFAR10_fedMedian.txt", sep=";", names=["time","evaltime","precision","recall","f1-score","loss_distributed","loss_centralized","accuracy_centralize","accuracy_distributed", "-",],)


fig, ax1 = plt.subplots()

ax1.grid(True,linestyle="dotted", color='grey')

left, bottom, width, height = [0.52, 0.13, 0.36, 0.25]
ax2 = fig.add_axes([left, bottom, width, height])

ax1.axhline(y = 20, color = 'tab:cyan', linestyle = 'dashdot', linewidth=1,) 
ax1.plot(gaussian_filter1d(FedAvg["accuracy_distributed"][:maxline] * 100, sigma=sigmaPlot),  ls="solid", linewidth=1,marker="p",markersize=4,)
ax1.plot(gaussian_filter1d(FedAvgLoss["accuracy_distributed"][:maxline] * 100, sigma=sigmaPlot), ls="solid", linewidth=1,marker="v",markersize=4,)
ax1.plot(gaussian_filter1d(FedACS["accuracy_distributed"][:maxline] * 100, sigma=sigmaPlot), ls="solid", linewidth=1,marker="^",markersize=4,)
ax1.plot(gaussian_filter1d(FedOpt["accuracy_distributed"][:maxline] * 100, sigma=sigmaPlot),  ls="solid", linewidth=1,marker=">",markersize=4,)
ax1.plot(gaussian_filter1d(FedYogi["accuracy_distributed"][:maxline] * 100, sigma=sigmaPlot),  ls="solid", linewidth=1,marker="<",markersize=4,)
ax1.plot(gaussian_filter1d(FedTrim["accuracy_distributed"][:maxline] * 100, sigma=sigmaPlot),  ls="solid", linewidth=1,marker="d",markersize=4,)
ax1.plot(gaussian_filter1d(FedMed["accuracy_distributed"][:maxline] * 100, sigma=sigmaPlot), color='tab:gray', ls="solid", linewidth=1,marker="h",markersize=4,)
ax1.plot(gaussian_filter1d(TSFed["accuracy_distributed"][:maxline] * 100, sigma=sigmaPlot),  ls="solid", linewidth=1,marker="o",markersize=4,)

ax1.set_xticks(np.arange(0, maxline+1, 5))
ax1.set_ylim(bottom=0)

ax1.set_title("Comparative - CIFAR10")
ax1.set_xlabel("Communication Rounds")
ax1.set_ylabel("Accuracy (%)")
ax1.legend(["Target Accuracy",
"FedAvg",
"FedAvgLoss",
"FedACS",
"FedOpt",
"FedYogi",
"FedTrim",
"FedMed",
"TSFed",],loc='upper left',fontsize="8")

ax2.grid(True,linestyle="dotted", color='grey')
ax2.xaxis.tick_top()

ax2.axhline(y = 20, color = 'tab:cyan', linestyle = 'dashdot', linewidth=1,) 
ax2.plot(gaussian_filter1d(FedAvg["accuracy_distributed"][:maxline] * 100, sigma=sigmaPlot),  ls="solid", linewidth=1,marker="p",markersize=5,)
ax2.plot(gaussian_filter1d(FedAvgLoss["accuracy_distributed"][:maxline] * 100, sigma=sigmaPlot), ls="solid", linewidth=1,marker="v",markersize=5,)
ax2.plot(gaussian_filter1d(FedACS["accuracy_distributed"][:maxline] * 100, sigma=sigmaPlot), ls="solid", linewidth=1,marker="^",markersize=5,)
ax2.plot(gaussian_filter1d(FedOpt["accuracy_distributed"][:maxline] * 100, sigma=sigmaPlot),  ls="solid", linewidth=1,marker=">",markersize=5,)
ax2.plot(gaussian_filter1d(FedYogi["accuracy_distributed"][:maxline] * 100, sigma=sigmaPlot),  ls="solid", linewidth=1,marker="<",markersize=5,)
ax2.plot(gaussian_filter1d(FedTrim["accuracy_distributed"][:maxline] * 100, sigma=sigmaPlot),  ls="solid", linewidth=1,marker="d",markersize=5,)
ax2.plot(gaussian_filter1d(FedMed["accuracy_distributed"][:maxline] * 100, sigma=sigmaPlot), color='tab:gray', ls="solid", linewidth=1,marker="h",markersize=5,)
ax2.plot(gaussian_filter1d(TSFed["accuracy_distributed"][:maxline] * 100, sigma=sigmaPlot),  ls="solid", linewidth=1,marker="o",markersize=5,)


ax2.set_xlim([maxline-10,maxline])
ax2.set_ylim([22,34])

ax2.tick_params(axis='both', which='major', labelsize=8)

ax2.set_xticks(np.arange(maxline-10,maxline, 2))
ax2.grid(True)


plt.savefig('tsfed_cifar10.png',bbox_inches='tight',  dpi=700)	#save image

plt.show()
