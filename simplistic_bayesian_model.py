# %%
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD

# %%
model = BayesianNetwork(
    [
        ("world_war_iii", "lock-in"),
        ("AGI_by_year_Y", "lock-in"),
        ("misalignment_by_AGI", "lock-in"),
        ("stable_totalitarianism_by_year_Y", "lock-in"),
    ]
)

# %%
