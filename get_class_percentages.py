import pandas as pd
import matplotlib.pyplot as plt
colors = ['#008080', '#0373d3', '#816cce', '#00ebeb', '#9fa6e9', '#2f1c91']
# Load the dataset from the CSV file
data = pd.read_csv('data/diabetes_012_health_indicators_BRFSS2015.csv')

# Calculate the count and percentage of each unique value in the Diabetes_012 column
class_counts = data['Diabetes_012'].value_counts()
total_entries = len(data)
class_percentages = (class_counts / total_entries) * 100

# Plot the percentages as a bar chart
plt.figure(figsize=(8, 5))
bars = plt.bar(class_percentages.index, class_percentages.values, color=colors)

# Add percentage labels on the bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.1f}%', ha='center', va='bottom')

# Customize the plot
plt.title('Percentage of Each Class in Diabetes_012', fontsize=16)
plt.xlabel('Diabetes Class (0: No Diabetes, 1: Pre-Diabetes, 2: Diabetes)', fontsize=12)
plt.ylabel('Percentage (%)', fontsize=12)
plt.xticks(class_percentages.index, labels=class_percentages.index.astype(int), fontsize=10)
plt.ylim(0, max(class_percentages.values) + 5)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show the plot
plt.tight_layout()
plt.show()
