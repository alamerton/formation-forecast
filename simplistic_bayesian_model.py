# %%
import numpy as np
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# %%
## Network with three edges, each of which influence lock-in

model = BayesianNetwork(
    [
        ("world_war_iii", "lock-in"),
        ("AGI", "lock-in"),
        ("misalignment", "lock-in"),
    ]
)

time_points = np.array([2030, 2055, 2080, 2105, 2130])

# %%
## Conditional probability distributions for each node

cpd_AGI_means = TabularCPD(
    variable="AGI",
    variable_card=5,
    values=[[0.2809], [0.6433], [0.7649], [0.8673], [0.9250]],
)

cpd_misalignment = TabularCPD(
    variable="misalignment",
    variable_card=1,
    values=[[0.5293]],
)

cpd_wwiii = TabularCPD(
    variable="world_war_iii",
    variable_card=5,
    values=[[0.3], [0.3144], [0.3861], [0.4579], [0.5297]],
)

# %%
## Variable elimination
model.add_cpds(cpd_AGI_means, cpd_misalignment, cpd_wwiii)

inference = VariableElimination(model)

p_lock_in = inference.query(["lock_in"])

print("Probability distribution of lock-in:")
print(p_lock_in)

# %%
