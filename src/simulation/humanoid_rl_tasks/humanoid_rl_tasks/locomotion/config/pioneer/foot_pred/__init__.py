# From leggedrobotics/foothold-prediction-model ("Watch Your Step: Anticipated Perception for Humanoid Locomotion",
# Heyrman et al., ETH Zurich), fpm/foot_pred/__init__.py, MIT License, Copyright (c) 2026 ETH Zurich. Unmodified.
from .labels import compute_strictly_future_contact_positions, detect_resets, foothold_labels, world_to_root
from .model import FootPredictor, NormalizedFootPredictor
from .supervisor import FootPredictionSupervisor, latest_foot_model
from .trainer import fit_foot_predictor
