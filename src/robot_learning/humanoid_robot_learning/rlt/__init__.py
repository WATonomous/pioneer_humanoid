"""RL Token (RLT, arXiv 2604.23073) on a frozen Pioneer-1, for the peg_insert critical phase.

A frozen base policy proposes action chunks; a small encoder squeezes its context tokens into one "RL token"; a TD3
actor-critic on that token edits the base's chunk during the hard part of the task (here: from "peg grasped, carried
to within 3 cm of the hole" to "inserted"). See train.py for the full pipeline and the deviations from the paper.
"""
