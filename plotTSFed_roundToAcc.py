import matplotlib.pyplot as plt
import numpy as np

# Sample data for four x-axis groups, each with three values
groups = ['FedAvg\n[10]','FedAvgLoss','FedACS\n[18]','FedOpt\n[110]', 'FedYogi\n[111]', 'FedTrimMedAvg\n[116]','FedMedian\n[116]','TSFed']
values = np.array([[50, 23, 21], [27, 9, 11], [10, 44, 9], [41, 33, 27], [46, 23, 27], [50, 50, 75], [50, 50, 75], [28, 14, 8]])

# Define the width of each bar
bar_width = 0.2

# Create an array of evenly spaced x-axis values for each group
x = np.arange(len(groups))

plt.rc('grid', linestyle="dotted", color='grey')
plt.grid(True)

# Create the bar plot
bars1 = plt.bar(x - bar_width, values[:, 0], width=bar_width, label='MNIST (75%)', hatch='///')
bars2 = plt.bar(x, values[:, 1], width=bar_width, label='FMNIST (60%)', hatch='++')
bars3 = plt.bar(x + bar_width, values[:, 2], width=bar_width, label='CIFAR10 (20%)', hatch='xx')

# plt.xticks(np.arange(0, 51, 5))
plt.rc('grid', linestyle="dotted", color='grey')
plt.grid(True)
plt.axhline(y = 50, color = 'tab:blue', linestyle = 'dashed', linewidth=1, label='Maximum Rounds') 
plt.axhline(y = 50, color = 'tab:orange', linestyle = 'dotted', linewidth=1, label='Maximum Rounds') 
plt.axhline(y = 75, color = 'tab:green', linestyle = 'dashdot', linewidth=1, label='Maximum Rounds') 



# Set x-axis labels and the legend
plt.xlabel('Strategy')
plt.xticks(x, groups, fontsize=9)
plt.legend(loc='upper left')


# Set labels for y-axis and the title
plt.ylabel('Rounds (t)')
plt.title('Number of rounds required to achieve target accuracy (%)')
plt.ylim(0,80)


# Function to add text labels to each bar
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        if height == 50 or height == 75:
          plt.annotate('n/a',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 8),  # 3 points vertical offset
                     textcoords="offset points",
                     ha='center', va='bottom', rotation=90)
        else:
          plt.annotate('{}'.format(height),
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 8),  # 3 points vertical offset
                     textcoords="offset points",
                     ha='center', va='bottom', rotation=90)


# Add labels to all bars
add_labels(bars1)
add_labels(bars2)
add_labels(bars3)

plt.savefig('fl_comparative_AcctoRounds.png',bbox_inches='tight',  dpi=700)	#save image

# Show the plot
plt.tight_layout()
plt.show()
