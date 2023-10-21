import pandas as pd
from pandas import read_csv
from scipy.ndimage.filters import gaussian_filter1d
import matplotlib.pyplot as plt  # this is used for the plot the graph
import numpy as np

name = "mnist"
maxline = 50
sigmaPlot = 0.75


adct_25_5 = pd.read_csv("ADCT/adct_"+name+"_thesis25_5.txt", sep=";", names=["time","evaltime","precision","recall","f1-score","loss_distributed","loss_centralized","accuracy_centralize","accuracy_distributed", "-",],)
adct_25_10 = pd.read_csv("ADCT/adct_"+name+"_thesis25_10.txt", sep=";", names=["time","evaltime","precision","recall","f1-score","loss_distributed","loss_centralized","accuracy_centralize","accuracy_distributed", "-",],)
adct_25_15 = pd.read_csv("ADCT/adct_"+name+"_thesis25_15.txt", sep=";", names=["time","evaltime","precision","recall","f1-score","loss_distributed","loss_centralized","accuracy_centralize","accuracy_distributed", "-",],)

adct_50_5 = pd.read_csv("ADCT/adct_"+name+"_thesis50_5.txt", sep=";", names=["time","evaltime","precision","recall","f1-score","loss_distributed","loss_centralized","accuracy_centralize","accuracy_distributed", "-",],)
adct_50_10 = pd.read_csv("ADCT/adct_"+name+"_thesis50_10.txt", sep=";", names=["time","evaltime","precision","recall","f1-score","loss_distributed","loss_centralized","accuracy_centralize","accuracy_distributed", "-",],)
adct_50_15 = pd.read_csv("ADCT/adct_"+name+"_thesis50_15.txt", sep=";", names=["time","evaltime","precision","recall","f1-score","loss_distributed","loss_centralized","accuracy_centralize","accuracy_distributed", "-",],)

adct_75_5 = pd.read_csv("ADCT/adct_"+name+"_thesis75_5.txt", sep=";", names=["time","evaltime","precision","recall","f1-score","loss_distributed","loss_centralized","accuracy_centralize","accuracy_distributed", "-",],)
adct_75_10 = pd.read_csv("ADCT/adct_"+name+"_thesis75_10.txt", sep=";", names=["time","evaltime","precision","recall","f1-score","loss_distributed","loss_centralized","accuracy_centralize","accuracy_distributed", "-",],)
adct_75_15 = pd.read_csv("ADCT/adct_"+name+"_thesis75_15.txt", sep=";", names=["time","evaltime","precision","recall","f1-score","loss_distributed","loss_centralized","accuracy_centralize","accuracy_distributed", "-",],)


fig, ax1 = plt.subplots()

ax1.grid(True,linestyle="dotted", color='grey')

left, bottom, width, height = [0.52, 0.18, 0.36, 0.37]
ax2 = fig.add_axes([left, bottom, width, height])

ax1.axhline(y = 75, color = 'tab:cyan', linestyle = 'dashdot', linewidth=1,) 
ax1.plot(gaussian_filter1d(adct_25_5["accuracy_distributed"][:maxline] * 100, sigma=sigmaPlot), ls="solid", linewidth=1,marker="d",markersize=4,)
ax1.plot(gaussian_filter1d(adct_25_10["accuracy_distributed"][:maxline] * 100, sigma=sigmaPlot),  ls="solid", linewidth=1,marker="p",markersize=4,)
ax1.plot(gaussian_filter1d(adct_25_15["accuracy_distributed"][:maxline] * 100, sigma=sigmaPlot), ls="solid", linewidth=1,marker="v",markersize=4,)
ax1.plot(gaussian_filter1d(adct_50_5["accuracy_distributed"][:maxline] * 100, sigma=sigmaPlot), ls="solid", linewidth=1,marker="d",markersize=4,)
ax1.plot(gaussian_filter1d(adct_50_10["accuracy_distributed"][:maxline] * 100, sigma=sigmaPlot),  ls="solid", linewidth=1,marker="p",markersize=4,)
ax1.plot(gaussian_filter1d(adct_50_15["accuracy_distributed"][:maxline] * 100, sigma=sigmaPlot),  ls="solid", linewidth=1,marker="v",markersize=4,)
ax1.plot(gaussian_filter1d(adct_75_5["accuracy_distributed"][:maxline] * 100, sigma=sigmaPlot),  ls="solid", linewidth=1,marker="d",markersize=4,)
ax1.plot(gaussian_filter1d(adct_75_10["accuracy_distributed"][:maxline] * 100, sigma=sigmaPlot),  ls="solid", linewidth=1,marker="p",markersize=4,)
ax1.plot(gaussian_filter1d(adct_75_15["accuracy_distributed"][:maxline] * 100, sigma=sigmaPlot),  ls="solid", linewidth=1,marker="v",markersize=4,)

ax1.set_xticks(np.arange(0, maxline+1, 5))

ax1.set_title("ADCT - MNIST")
ax1.set_xlabel("Communication Rounds")
ax1.set_ylabel("Accuracy (%)")
ax1.legend(["Target Accuracy","ADCT(25,5)","ADCT(25,10)","ADCT(25,15)","ADCT(50,5)","ADCT(50,10)","ADCT(50,15)","ADCT(75,5)","ADCT(75,10)","ADCT(75,15)"],loc='upper left',fontsize="8")

ax2.grid(True,linestyle="dotted", color='grey')
ax2.xaxis.tick_top()

ax2.axhline(y = 75, color = 'tab:cyan', linestyle = 'dashdot', linewidth=1,) 
ax2.plot(gaussian_filter1d(adct_25_5["accuracy_distributed"][:maxline] * 100, sigma=sigmaPlot), ls="solid", linewidth=1,marker="d",markersize=5,)
ax2.plot(gaussian_filter1d(adct_25_10["accuracy_distributed"][:maxline] * 100, sigma=sigmaPlot),  ls="solid", linewidth=1,marker="p",markersize=5,)
ax2.plot(gaussian_filter1d(adct_25_15["accuracy_distributed"][:maxline] * 100, sigma=sigmaPlot), ls="solid", linewidth=1,marker="v",markersize=5,)
ax2.plot(gaussian_filter1d(adct_50_5["accuracy_distributed"][:maxline] * 100, sigma=sigmaPlot), ls="solid", linewidth=1,marker="d",markersize=5,)
ax2.plot(gaussian_filter1d(adct_50_10["accuracy_distributed"][:maxline] * 100, sigma=sigmaPlot),  ls="solid", linewidth=1,marker="p",markersize=5,)
ax2.plot(gaussian_filter1d(adct_50_15["accuracy_distributed"][:maxline] * 100, sigma=sigmaPlot),  ls="solid", linewidth=1,marker="v",markersize=5,)
ax2.plot(gaussian_filter1d(adct_75_5["accuracy_distributed"][:maxline] * 100, sigma=sigmaPlot),  ls="solid", linewidth=1,marker="d",markersize=5,)
ax2.plot(gaussian_filter1d(adct_75_10["accuracy_distributed"][:maxline] * 100, sigma=sigmaPlot),  ls="solid", linewidth=1,marker="p",markersize=5,)
ax2.plot(gaussian_filter1d(adct_75_15["accuracy_distributed"][:maxline] * 100, sigma=sigmaPlot),  ls="solid", linewidth=1,marker="v",markersize=5,)

ax2.set_xlim([maxline-10,maxline])
ax2.set_ylim([73,81])

ax2.tick_params(axis='both', which='major', labelsize=8)

ax2.set_xticks(np.arange(maxline-10,maxline, 2))
ax2.grid(True)


plt.savefig('adct_'+name+'.png',bbox_inches='tight',  dpi=700)	#save image

plt.show()
