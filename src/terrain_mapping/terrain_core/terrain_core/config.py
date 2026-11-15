"""Size of the policy map: the height map sent to the locomotion policy.

The numbers are in config/terrain.yaml.
"""
from dataclasses import dataclass


@dataclass
class PolicyMapConfig:
    length_m: float      # forward
    width_m: float       # side to side
    resolution_m: float  # cell size

    @classmethod
    def from_yaml(cls, path):
        import yaml
        with open(path) as fh:
            settings = yaml.safe_load(fh) or {}
        return cls(**settings["policy"])

    def shape(self):
        """Number of cells as (forward, sideways)."""
        return round(self.length_m / self.resolution_m), round(self.width_m / self.resolution_m)
