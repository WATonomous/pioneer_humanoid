"""Human authority and correction boundaries, independent of the simulator/UI."""


class HumanGate:
    def __init__(self, env, expert, actor, save_segment, max_frames):
        self.env, self.expert, self.actor = env, expert, actor
        self.save_segment, self.max_frames = save_segment, max_frames
        self.mode = "POLICY" if actor else "HUMAN"
        self.paused = False
        self.frames = []

    def toggle(self):
        if self.actor is None:
            return
        if self.mode == "POLICY":
            self.actor.reset()
            self.expert.seed(self.env)
            self.mode = "HUMAN"
        else:
            self.save("hand_back")
            self.actor.reset()
            self.mode = "POLICY"

    def pause(self):
        self.paused = not self.paused
        if self.actor:
            self.actor.reset()
        self.expert.seed(self.env)

    def reset(self):
        self.frames.clear()
        if self.actor:
            self.actor.reset()
        self.expert.seed(self.env)

    def select(self, obs, task):
        if self.paused:
            raise RuntimeError("Cannot select an action while paused")
        return self.expert.action(self.env) if self.mode == "HUMAN" else self.actor.action(obs, task)

    def record(self, obs, executed_action, task):
        # This is the sole training-label admission point. Policy execution can
        # never enter the expert dataset, even if a caller invokes record().
        if self.mode != "HUMAN" or self.paused:
            return
        self.frames.append({**obs, "action": executed_action, "task": task})
        if len(self.frames) >= self.max_frames:
            self.save("buffer_boundary")

    def save(self, reason):
        if self.frames:
            self.save_segment(self.frames, reason)
            self.frames = []
