import numpy as np
import matplotlib.pyplot as plt

# Data
data = {
    100: {'No LDA': 0.8393772171856524, 1: 0.780449349625542, 2: 0.8242018131651557, 3: 0.8383918013401656, 4: 0.8379976350019709, 5: 0.8381947181710682}
}

components = list(data.keys())
categories = list(data[100].keys())

# Prepare data for plotting
x = np.arange(len(components))  # X-axis positions
width = 0.15  # Bar width

# Create a bar for each category
fig, ax = plt.subplots(figsize=(10, 6))
for i, category in enumerate(categories):
    values = [data[comp][category] for comp in components]
    ax.bar(x + i * width, values, width, label=category)

# Add labels, legend, and title
ax.set_xlabel('Number of Components')
ax.set_ylabel('Accuracy')
ax.set_title('Performance across LDA Components')
ax.set_xticks(x + width * (len(categories) - 1) / 2)
ax.set_xticklabels(components)
ax.legend(title='Category')

# Set y-axis limit with a higher minimum value
ax.set_ylim(0.75, 0.85)  # Adjust minimum and maximum as needed
ax.grid(axis='y')

# Show plot
plt.show()
