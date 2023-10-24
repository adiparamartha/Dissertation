import pandas as pd
from pandas import read_csv
from scipy.ndimage.filters import gaussian_filter1d
import matplotlib.pyplot as plt  # this is used for the plot the graph
import numpy as np

maxline = 50
sigmaPlot = 0.75


TSFed = pd.read_csv("TSFed/tsfed_mnist_thesis.txt", sep=";", names=["time","evaltime","precision","recall","f1-score","loss_distributed","loss_centralized","accuracy_centralize","accuracy_distributed", "-",],)
FedAvg = pd.read_csv("Baseline/baseline_MNIST_fedAvg.txt", sep=";", names=["time","evaltime","precision","recall","f1-score","loss_distributed","loss_centralized","accuracy_centralize","accuracy_distributed", "-",],)
FedAvgLoss = pd.read_csv("Baseline/baseline_MNIST_fedAvgLoss.txt", sep=";", names=["time","evaltime","precision","recall","f1-score","loss_distributed","loss_centralized","accuracy_centralize","accuracy_distributed", "-",],)
FedACS = pd.read_csv("Baseline/baseline_MNIST_fedAvgAcc.txt", sep=";", names=["time","evaltime","precision","recall","f1-score","loss_distributed","loss_centralized","accuracy_centralize","accuracy_distributed", "-",],)
FedOpt = pd.read_csv("Baseline/baseline_MNIST_fedOpt.txt", sep=";", names=["time","evaltime","precision","recall","f1-score","loss_distributed","loss_centralized","accuracy_centralize","accuracy_distributed", "-",],)
FedYogi = pd.read_csv("Baseline/baseline_MNIST_fedYogi.txt", sep=";", names=["time","evaltime","precision","recall","f1-score","loss_distributed","loss_centralized","accuracy_centralize","accuracy_distributed", "-",],)
FedTrim = pd.read_csv("Baseline/baseline_MNIST_fedTrimMedAvg.txt", sep=";", names=["time","evaltime","precision","recall","f1-score","loss_distributed","loss_centralized","accuracy_centralize","accuracy_distributed", "-",],)
FedMed = pd.read_csv("Baseline/baseline_MNIST_fedMedian.txt", sep=";", names=["time","evaltime","precision","recall","f1-score","loss_distributed","loss_centralized","accuracy_centralize","accuracy_distributed", "-",],)


fig, ax1 = plt.subplots()

ax1.grid(True,linestyle="dotted", color='grey')

left, bottom, width, height = [0.52, 0.13, 0.36, 0.30]
ax2 = fig.add_axes([left, bottom, width, height])

ax1.axhline(y = 75, color = 'tab:cyan', linestyle = 'dashdot', linewidth=1,) 
ax1.plot(gaussian_filter1d(FedAvg["accuracy_distributed"][:maxline] * 100, sigma=sigmaPlot),  ls="solid", linewidth=1,marker="p",markersize=4,)
ax1.plot(gaussian_filter1d(FedAvgLoss["accuracy_distributed"][:maxline] * 100, sigma=sigmaPlot), ls="solid", linewidth=1,marker="v",markersize=4,)
ax1.plot(gaussian_filter1d(FedACS["accuracy_distributed"][:maxline] * 100, sigma=sigmaPlot), ls="solid", linewidth=1,marker="^",markersize=4,)
ax1.plot(gaussian_filter1d(FedOpt["accuracy_distributed"][:maxline] * 100, sigma=sigmaPlot),  ls="solid", linewidth=1,marker=">",markersize=4,)
ax1.plot(gaussian_filter1d(FedYogi["accuracy_distributed"][:maxline] * 100, sigma=sigmaPlot),  ls="solid", linewidth=1,marker="<",markersize=4,)
ax1.plot(gaussian_filter1d(FedTrim["accuracy_distributed"][:maxline] * 100, sigma=sigmaPlot),  ls="solid", linewidth=1,marker="d",markersize=4,)
ax1.plot(gaussian_filter1d(FedMed["accuracy_distributed"][:maxline] * 100, sigma=sigmaPlot), color='tab:gray', ls="solid", linewidth=1,marker="h",markersize=4,)
ax1.plot(gaussian_filter1d(TSFed["accuracy_distributed"][:maxline] * 100, sigma=sigmaPlot),  ls="solid", linewidth=1,marker="o",markersize=4,)

ax1.set_xticks(np.arange(0, maxline+1, 5))
ax1.set_ylim(bottom=10)

ax1.set_title("Comparative - MNIST")
ax1.set_xlabel("Communication Rounds")
ax1.set_ylabel("Accuracy (%)")
ax1.legend(["Target Accuracy",
"FedAvg [10]",
"FedAvgLoss",
"FedACS [18]",
"FedOpt [110]",
"FedYogi [111]",
"FedTrim [116]",
"FedMed [116]",
"TSFed",],loc='upper left',fontsize="8")

ax2.grid(True,linestyle="dotted", color='grey')
ax2.xaxis.tick_top()

ax2.axhline(y = 75, color = 'tab:cyan', linestyle = 'dashdot', linewidth=1,) 
ax2.plot(gaussian_filter1d(FedAvg["accuracy_distributed"][:maxline] * 100, sigma=sigmaPlot),  ls="solid", linewidth=1,marker="p",markersize=5,)
ax2.plot(gaussian_filter1d(FedAvgLoss["accuracy_distributed"][:maxline] * 100, sigma=sigmaPlot), ls="solid", linewidth=1,marker="v",markersize=5,)
ax2.plot(gaussian_filter1d(FedACS["accuracy_distributed"][:maxline] * 100, sigma=sigmaPlot), ls="solid", linewidth=1,marker="^",markersize=5,)
ax2.plot(gaussian_filter1d(FedOpt["accuracy_distributed"][:maxline] * 100, sigma=sigmaPlot),  ls="solid", linewidth=1,marker=">",markersize=5,)
ax2.plot(gaussian_filter1d(FedYogi["accuracy_distributed"][:maxline] * 100, sigma=sigmaPlot),  ls="solid", linewidth=1,marker="<",markersize=5,)
ax2.plot(gaussian_filter1d(FedTrim["accuracy_distributed"][:maxline] * 100, sigma=sigmaPlot),  ls="solid", linewidth=1,marker="d",markersize=5,)
ax2.plot(gaussian_filter1d(FedMed["accuracy_distributed"][:maxline] * 100, sigma=sigmaPlot), color='tab:gray', ls="solid", linewidth=1,marker="h",markersize=5,)
ax2.plot(gaussian_filter1d(TSFed["accuracy_distributed"][:maxline] * 100, sigma=sigmaPlot),  ls="solid", linewidth=1,marker="o",markersize=5,)


ax2.set_xlim([maxline-10,maxline])
ax2.set_ylim([61,88])

ax2.tick_params(axis='both', which='major', labelsize=8)

ax2.set_xticks(np.arange(maxline-10,maxline, 2))
ax2.grid(True)


plt.savefig('tsfed_mnist.png',bbox_inches='tight',  dpi=700)	#save image

plt.show()
