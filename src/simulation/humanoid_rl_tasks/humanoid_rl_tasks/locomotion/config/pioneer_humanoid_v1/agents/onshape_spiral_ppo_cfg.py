"""PPO defaults for the Onshape spiral staircase task.

Same as the stairs task except for one thing: the entropy bonus is off.

The entropy coefficient is what makes a policy *raise* its action noise when
it is not making progress, and on a staircase the extra noise shows up as
thrashing -- the joint targets swing by a radian and the robot falls off the
side of the stairs (the run log showed the action noise std growing 3.7 ->
4.0 -> 4.9 while the out-of-bounds rate sat at 54%). The straight course
tolerates it because its treads are 0.28 m deep and flat; the spiral's are
0.42 m deep but only 0.12 m high, and a robot that lurches 0.4 m sideways
steps off them.

With the coefficient at zero the noise stays where the warm start left it
(3.7) and only shrinks if the policy actually improves.
"""

from isaaclab.utils import configclass

from .stairs_ppo_cfg import PioneerHumanoidStairsPPORunnerCfg


@configclass
class PioneerHumanoidOnshapeSpiralPPORunnerCfg(PioneerHumanoidStairsPPORunnerCfg):
    """The stairs runner with the exploration noise frozen."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "pioneer_humanoid_spiral"
        self.algorithm.entropy_coef = 0.0
