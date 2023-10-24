import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter

# Sample data for four x-axis groups, each with three values
groups = ['FedAvg\n[10]','FedAvgLoss','FedACS\n[18]','FedOpt\n[110]', 'FedYogi\n[111]', 'FedTrimMedAvg\n[116]','FedMedian\n[116]','TSFed']
values = np.array([[475994.9655532837, 532993.8404560089, 928548.8657951355], [477341.4304256439, 536950.2220153809, 933159.2531204224], [472774.2714881897, 536722.0914363861, 932524.7666835785], [470800.7357120514, 534793.133020401, 938473.6092090607], [473799.4203567505, 534687.2634887695, 929035.478591919], [475144.0780162811, 534691.4267539978, 935101.539850235], [473050.35281181335, 537286.7579460144, 934425.0483512878], [669748.663187027, 680039.31117057, 1010842.4615859985]])

values_in_minutes = values / 60000.0

# Define the width of each bar
bar_width = 0.2

# Create an array of evenly spaced x-axis values for each group
x = np.arange(len(groups))

plt.rc('grid', linestyle="dotted", color='grey')
plt.grid(True)

# Create the bar plot
bars1 = plt.bar(x - bar_width, values_in_minutes[:, 0], width=bar_width, label='MNIST', hatch='///')
bars2 = plt.bar(x, values_in_minutes[:, 1], width=bar_width, label='FMNIST', hatch='++')
bars3 = plt.bar(x + bar_width, values_in_minutes[:, 2], width=bar_width, label='CIFAR10', hatch='xx')

# Set x-axis labels and the legend
plt.ylim(0,20)
plt.xlabel('Strategy')
plt.xticks(x, groups, fontsize=9)
plt.legend(loc='upper left')

# Set labels for y-axis and the title
plt.ylabel('Time (minutes)')
plt.title('Overall training time of FL process')

# Set y-axis labels in scientific notation
ax = plt.gca()
ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True, useOffset=False))

# Function to add text labels to each bar
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        if height == 50 or height == 75:
            plt.annotate('n/a',
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 8),  # 3 points vertical offset
                         textcoords="offset points",
                         ha='center', va='bottom', rotation=90, fontsize=10)
        else:
            plt.annotate('{:.2f}'.format(height),
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 8),  # 3 points vertical offset
                         textcoords="offset points",
                         ha='center', va='bottom', rotation=90, fontsize=10)
            # plt.text(bar.get_x() + bar.get_width() / 2, height + 1, '{:.2f}'.format(height), fontsize=8,
            #          ha='center', va='bottom', rotation=90, bbox=dict(facecolor='white', edgecolor='white'))

# Add labels to all bars
add_labels(bars1)
add_labels(bars2)
add_labels(bars3)

plt.savefig('fl_comparative_trainingTime.png', bbox_inches='tight', dpi=700)  # Save image

# Show the plot
plt.tight_layout()
plt.show()
