# %%
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD

# %%
## Network with three edges, each of which influence lock-in

model = BayesianNetwork(
    [
        ("world_war_iii", "lock-in"),
        ("AGI", "lock-in"),
        ("misalignment", "lock-in"),
    ]
)

# %%
## Conditional probability distributions for each node

cpd_AGI_means = TabularCPD(
    variable="AGI",
    variable_card=5,
    values=[[0.2809, 0.6433, 0.7649, 0.8673, 0.9250]],
)

cpd_misalignment = TabularCPD(
    variable="misalignment",
    variable_card=2,
    values=[[0.5293]],
)

cpd_wwiii = TabularCPD(
    variable="world_war_iii",
    variable_card=5,
    values=[[0.3, 0.3144, 0.3861, 0.4579, 0.5297]],
)

# %%
## Conditional probability distributions for lock-in
cpd_lock_in = TabularCPD(
    variable="lock_in",
    variable_card=2,
    values=[
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
    ],
    evidence=["world_war_iii", "AGI", "misalignment"],
    evidence_card=[5, 5, 2],
)

# %%
model.add_cpds(cpd_AGI_means, cpd_misalignment, cpd_wwiii, cpd_lock_in)

# %%
print(model.check_model())
