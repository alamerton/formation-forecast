# %%
import numpy as np

# Define the arrays
arrays = [
    [0.08, 0.297, 0.432, 0.567, 0.702],  # Epoch model-based
    [0.12, 0.601, 0.756, 0.911, 1.0],  # Epoch judgement-based
    [0.6572, 0.9334, 0.9603, 0.9697, 0.9745],  # Metaculus weakly general
    [0.4183, 0.8478, 0.9002, 0.9279, 0.9552],  # Metaculus general
    [0.1, 0.5324, 0.8027, 1.0, 1.0],  # AI Impacts HLMI
    [0.31, 0.648, 0.738, 0.828, 0.918],  # Samotsvety
]

# %%

# Calculate and print the mean for each array
# for i, arr in enumerate(arrays, 1):
#     mean = np.mean(arr)
#     print(f"Mean of array {i}: {mean:.4f}")

# %%
# Convert the list of lists to a numpy array
np_arrays = np.array(arrays)

# Calculate the average for each index
averages = np.mean(np_arrays, axis=0)

# Print the results
for i, avg in enumerate(averages):
    print(f"Average for index {i}: {avg:.4f}")

# %%
