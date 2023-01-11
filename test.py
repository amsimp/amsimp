# Import dependencies.
import amsimp
import iris

# Deterministic.
print("Deterministic")

# Define state.
state = amsimp.OperationalModel()

# Generate forecast and save.
state.generate_forecast(save=True)

# Ensemble.
print("Ensemble")

# Define state.
state = amsimp.OperationalModel(use_efs=True)

# Generate forecast and save.
state.generate_forecast(save=True)
