import matplotlib.pyplot as plt
import numpy as np

# Data for two models and four metrics
model1_metrics = [10, 15, 20, 25]  # Metrics for model 1
model2_metrics = [12, 17, 22, 27]  # Metrics for model 2
metrics = ['Metric 1', 'Metric 2', 'Metric 3', 'Metric 4']  # Metric names

# Set the width of the bars
bar_width = 0.35

# Set the positions of the bars on the x-axis
x = np.arange(len(metrics))

# Plotting the bars for model 1
plt.bar(x - bar_width/2, model1_metrics, bar_width, label='Model 1')

# Plotting the bars for model 2
plt.bar(x + bar_width/2, model2_metrics, bar_width, label='Model 2')

# Adding labels, title, and ticks
plt.xlabel('Metrics')
plt.ylabel('Scores')
plt.title('Comparison of Two Models')
plt.xticks(x, metrics)
plt.legend()

# Show plot
plt.show()
