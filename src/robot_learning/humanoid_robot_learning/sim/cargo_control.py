"""Human authority and correction boundaries, independent of the simulator/UI."""


class HumanGate:
    def __init__(self, env, expert, actor, save_segment, max_frames, save_on_hand_back=True):
        self.env, self.expert, self.actor = env, expert, actor
        self.save_segment, self.max_frames = save_segment, max_frames
        self.mode = "POLICY" if actor else "HUMAN"
        self.paused = False
        self.frames = []
        self.pending_segments = []
        self.save_on_hand_back = save_on_hand_back
        self.notice = ""
        self.saved_segments = 0

    @property
    def pending_frames(self):
        return len(self.frames) + sum(len(frames) for frames in self.pending_segments)

    @property
    def buffer_full(self):
        return not self.save_on_hand_back and self.pending_frames >= self.max_frames

    def _stage(self):
        if self.frames:
            self.pending_segments.append(self.frames)
            self.frames = []

    def discard(self):
        count = self.pending_frames
        self.frames.clear()
        self.pending_segments.clear()
        self.notice = ""
        return count

    def toggle(self):
        if self.actor is None:
            return
        if self.mode == "POLICY":
            self.notice = ""
            self.actor.reset()
            self.expert.seed(self.env)
            self.mode = "HUMAN"
        else:
            if self.save_on_hand_back:
                self.save("hand_back")
            else:
                self._stage()  # keep discontinuous interventions separate until explicitly accepted
            self.actor.reset()
            self.mode = "POLICY"

    def pause(self):
        self.paused = not self.paused
        if self.actor:
            self.actor.reset()
        self.expert.seed(self.env)

    def reset(self, resume_policy=False):
        self.discard()
        if resume_policy:
            self.mode = "POLICY" if self.actor else "HUMAN"
        if self.actor:
            self.actor.reset()
        self.expert.seed(self.env)

    def select(self, obs, task):
        if self.paused or self.buffer_full:
            raise RuntimeError("Cannot select an action while paused or awaiting a save/discard decision")
        return self.expert.action(self.env) if self.mode == "HUMAN" else self.actor.action(obs, task)

    def record(self, obs, executed_action, task):
        # This is the sole training-label admission point. Policy execution can
        # never enter the expert dataset, even if a caller invokes record().
        if self.mode != "HUMAN" or self.paused or self.buffer_full:
            return
        self.frames.append({**obs, "action": executed_action, "task": task})
        if self.save_on_hand_back and len(self.frames) >= self.max_frames:
            self.save("buffer_boundary")

    def save(self, reason):
        self._stage()
        while self.pending_segments:
            self.save_segment(self.pending_segments[0], reason)
            self.saved_segments += 1
            self.pending_segments.pop(0)
