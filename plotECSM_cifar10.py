import pandas as pd
from pandas import read_csv
from scipy.ndimage.filters import gaussian_filter1d
import matplotlib.pyplot as plt  # this is used for the plot the graph
import numpy as np

name = "CIFAR10"
maxline = 75
sigmaPlot = 0.75


mnist_1_8 = pd.read_csv("ECSM/ECSM"+name+"_1_8.txt", sep=";", names=["time","evaltime","precision","recall","f1-score","loss_distributed","loss_centralized","accuracy_centralize","accuracy_distributed", "-",],)
mnist_2_7 = pd.read_csv("ECSM/ECSM"+name+"_2_7.txt", sep=";", names=["time","evaltime","precision","recall","f1-score","loss_distributed","loss_centralized","accuracy_centralize","accuracy_distributed", "-",],)
mnist_3_6 = pd.read_csv("ECSM/ECSM"+name+"_3_6.txt", sep=";", names=["time","evaltime","precision","recall","f1-score","loss_distributed","loss_centralized","accuracy_centralize","accuracy_distributed", "-",],)
mnist_4_5 = pd.read_csv("ECSM/ECSM"+name+"_4_5.txt", sep=";", names=["time","evaltime","precision","recall","f1-score","loss_distributed","loss_centralized","accuracy_centralize","accuracy_distributed", "-",],)
mnist_5_4 = pd.read_csv("ECSM/ECSM"+name+"_5_4.txt", sep=";", names=["time","evaltime","precision","recall","f1-score","loss_distributed","loss_centralized","accuracy_centralize","accuracy_distributed", "-",],)
mnist_6_3 = pd.read_csv("ECSM/ECSM"+name+"_6_3.txt", sep=";", names=["time","evaltime","precision","recall","f1-score","loss_distributed","loss_centralized","accuracy_centralize","accuracy_distributed", "-",],)
mnist_7_2 = pd.read_csv("ECSM/ECSM"+name+"_7_2.txt", sep=";", names=["time","evaltime","precision","recall","f1-score","loss_distributed","loss_centralized","accuracy_centralize","accuracy_distributed", "-",],)
mnist_8_1 = pd.read_csv("ECSM/ECSM"+name+"_8_1.txt", sep=";", names=["time","evaltime","precision","recall","f1-score","loss_distributed","loss_centralized","accuracy_centralize","accuracy_distributed", "-",],)


fig, ax1 = plt.subplots()

ax1.grid(True,linestyle="dotted", color='grey')

left, bottom, width, height = [0.52, 0.13, 0.36, 0.30]
ax2 = fig.add_axes([left, bottom, width, height])

ax1.axhline(y = 20, color = 'tab:cyan', linestyle = 'dashdot', linewidth=1,) 
ax1.plot(gaussian_filter1d(mnist_1_8["accuracy_distributed"][:maxline] * 100, sigma=sigmaPlot), ls="solid", linewidth=1,marker="o",markersize=4,)
ax1.plot(gaussian_filter1d(mnist_2_7["accuracy_distributed"][:maxline] * 100, sigma=sigmaPlot),  ls="solid", linewidth=1,marker="o",markersize=4,)
ax1.plot(gaussian_filter1d(mnist_3_6["accuracy_distributed"][:maxline] * 100, sigma=sigmaPlot), ls="solid", linewidth=1,marker="o",markersize=4,)
ax1.plot(gaussian_filter1d(mnist_4_5["accuracy_distributed"][:maxline] * 100, sigma=sigmaPlot), ls="solid", linewidth=1,marker="o",markersize=4,)
ax1.plot(gaussian_filter1d(mnist_5_4["accuracy_distributed"][:maxline] * 100, sigma=sigmaPlot),  ls="solid", linewidth=1,marker="o",markersize=4,)
ax1.plot(gaussian_filter1d(mnist_6_3["accuracy_distributed"][:maxline] * 100, sigma=sigmaPlot),  ls="solid", linewidth=1,marker="o",markersize=4,)
ax1.plot(gaussian_filter1d(mnist_7_2["accuracy_distributed"][:maxline] * 100, sigma=sigmaPlot),  ls="solid", linewidth=1,marker="o",markersize=4,)
ax1.plot(gaussian_filter1d(mnist_8_1["accuracy_distributed"][:maxline] * 100, sigma=sigmaPlot),  ls="solid", linewidth=1,marker="o",markersize=4,)

ax1.set_xticks(np.arange(0, maxline+1, 5))
ax1.set_ylim(bottom=5)

ax1.set_title("ECSM - CIFAR10")
ax1.set_xlabel("Communication Rounds")
ax1.set_ylabel("Accuracy (%)")
ax1.legend(["Target Accuracy","ECSM(0.1,0.8,0.1)","ECSM(0.2,0.7,0.1)","ECSM(0.3,0.6,0.1)","ECSM(0.4,0.5,0.1)","ECSM(0.5,0.4,0.1)","ECSM(0.6,0.3,0.1)","ECSM(0.7,0.2,0.1)","ECSM(0.8,0.1,0.1)"],loc='upper left',fontsize="8")

ax2.grid(True,linestyle="dotted", color='grey')
ax2.xaxis.tick_top()

ax2.axhline(y = 20, color = 'tab:cyan', linestyle = 'dashdot', linewidth=1,) 
ax2.plot(gaussian_filter1d(mnist_1_8["accuracy_distributed"][:maxline] * 100, sigma=sigmaPlot), ls="solid", linewidth=1,marker="o",markersize=5,)
ax2.plot(gaussian_filter1d(mnist_2_7["accuracy_distributed"][:maxline] * 100, sigma=sigmaPlot),  ls="solid", linewidth=1,marker="o",markersize=5,)
ax2.plot(gaussian_filter1d(mnist_3_6["accuracy_distributed"][:maxline] * 100, sigma=sigmaPlot), ls="solid", linewidth=1,marker="o",markersize=5,)
ax2.plot(gaussian_filter1d(mnist_4_5["accuracy_distributed"][:maxline] * 100, sigma=sigmaPlot), ls="solid", linewidth=1,marker="o",markersize=5,)
ax2.plot(gaussian_filter1d(mnist_5_4["accuracy_distributed"][:maxline] * 100, sigma=sigmaPlot),  ls="solid", linewidth=1,marker="o",markersize=5,)
ax2.plot(gaussian_filter1d(mnist_6_3["accuracy_distributed"][:maxline] * 100, sigma=sigmaPlot),  ls="solid", linewidth=1,marker="o",markersize=5,)
ax2.plot(gaussian_filter1d(mnist_7_2["accuracy_distributed"][:maxline] * 100, sigma=sigmaPlot),  ls="solid", linewidth=1,marker="o",markersize=5,)
ax2.plot(gaussian_filter1d(mnist_8_1["accuracy_distributed"][:maxline] * 100, sigma=sigmaPlot),  ls="solid", linewidth=1,marker="o",markersize=5,)

ax2.set_xlim([maxline-10,maxline])
ax2.set_ylim([30,36])

ax2.tick_params(axis='both', which='major', labelsize=8)

ax2.set_xticks(np.arange(maxline-10,maxline, 2))
ax2.grid(True)

plt.savefig('ecsm_'+name.lower()+'.png',bbox_inches='tight',  dpi=700)	#save image

plt.show()
