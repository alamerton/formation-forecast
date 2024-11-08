# %%
import matplotlib.pyplot as plt
import numpy as np

# Generate some random data
data = []

# Create the histogram
plt.hist(data, bins=30)

# Add labels and title
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.title("Simple Histogram")

# Display the plot
plt.show()
# %%
