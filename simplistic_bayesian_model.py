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
        # ("stable_totalitarianism", "lock-in"),
        # ("epoch_model", "AGI"),
        # ("epoch_judgement", "AGI"),
        # ("metaculus_weak", "AGI"),
        # ("metaculus_general", "AGI"),
        # ("AI_impacts", "AGI"),
        # ("samotsvety", "AGI"),
    ]
)

# %%
## Conditional probability distributions for each node

# Epoch model-based
# cpd_epoch_model = TabularCPD(
#     variable="epoch_model", variable_card=5, values=[[0.08, 0.297, 0.432, 0.567, 0.702]]
# )

# Epoch judgement-based
# cpd_epoch_judgement = TabularCPD(
#     variable="epoch_judgement",
#     variable_card=5,
#     values=[[0.12, 0.601, 0.756, 0.911, 1.0]],
# )

# Metaculus weakly general
# cpd_metaculus_weak = TabularCPD(
#     variable="metaulus_weak",
#     variable_card=5,
#     values=[[0.6572, 0.9334, 0.9603, 0.9697, 0.9745]],
# )

# Metaculus general
# cpd_metaculus_general = TabularCPD(
#     variable="metaculus_general",
#     variable_card=5,
#     values=[[0.4183, 0.8478, 0.9002, 0.9279, 0.9552]],
# )

# AI Impacts HLMI
# cpd_AI_impacts = TabularCPD(
#     variable="AI_impacts", variable_card=5, values=[[0.1, 0.5324, 0.8027, 1.0, 1.0]]
# )

# Samotsvety
# cpd_samotsvety = TabularCPD(
#     variable="samotsvety", variable_card=5, values=[[0.31, 0.648, 0.738, 0.828, 0.918]]
# )

cpb_AGI_means = TabularCPD(
    variable="AGI",
    variable_card=5,
    values=[[0.4156, 0.6776, 0.8990, 0.8099, 0.6870, 0.6884]],
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
cpd_lock_in = TabularCPD
