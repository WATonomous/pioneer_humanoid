"""Loop closure detector: appearance retrieval -> geometric verification -> plausibility gate ->
confirmation over several keyframes -> pairwise consistency.

It sits next to the backend, not inside VO. Feed it every keyframe as it is created (the Keyframe plus one
global descriptor for its image). It returns a LoopUpdate: loop edges to add and, because a later loop can show an
earlier one was wrong, ids of loop edges to take back out. Nothing here needs ROS, GTSAM, OpenCV or a GPU: the
verifier is passed in.

The stages follow what ORB-SLAM3, VINS, RTAB-Map and Kimera do (see my_notes/loop_closure_research.md):
retrieval by appearance (not by the drifted position), recent keyframes excluded by odometry hops, every
candidate verified and the best kept, a gate against odometry, and agreement over consecutive keyframes.
"""
from dataclasses import dataclass


from .consistency import best_consistent_set, consistent
from .gate import CHI2_6_999, plausibility
from .graph import KeyframeGraph
from .retrieval import DescriptorDatabase
from .types import LoopConstraint, LoopUpdate, Rejection


# Every knob in one place. The defaults were tuned on TUM RGB-D (fr1/room, fr2/desk, fr3/long_office, then checked on
# fr1/desk, fr1/desk2, fr1/360) and checked for false loops in the simulators, NOT on D455 data. Re-sweep them on
# labelled recordings with slam_tools/eval_detector.py
@dataclass
class LoopParams:
    min_keyframes: int = 12         # don't look for loops before this many keyframes exist (ORB-SLAM3 uses 12)
    min_hops: int = 20              # a candidate must be at least this many odometry edges from the current keyframe
    min_path: float = 2.0           # ... and at least this far along the odometry path, in metres (so standing still makes no loops)
    top_k: int = 3                  # candidates taken from place recognition (each one costs a verification)
    min_score: float = 0.3          # similarity floor for a candidate (bag of words, tf-idf cosine)
    min_inliers: int = 30           # a verified loop needs at least this many inliers
    gate: bool = True               # reject loops that odometry says are implausible
    gate_inflation: float = 1.0     # >1 loosens the gate, for odometry with over-confident covariances
    gate_chi2: float = CHI2_6_999
    confirm_count: int = 2          # consecutive keyframes that must agree before a loop is accepted (1 = off)
    confirm_window: int = 4         # a supporting loop may come from this many keyframes back
    confirm_hops: int = 8           # supporting loops must point at old keyframes this close together
    pcm: bool = True                # keep only the largest mutually consistent group of loops
    pcm_chi2: float = CHI2_6_999
    max_loops_per_kf: int = 1       # best loops kept per keyframe
    # Keyframes to wait after accepting a loop: neighbouring loops add no new information, and many near-identical
    # edges make the graph over-confident
    cooldown: int = 10


class LoopClosureDetector:
    # verifier(old_id, new_id) -> Verification | None, see verification.py
    def __init__(self, verifier, params=None):
        self.p = params or LoopParams()
        self.verifier = verifier
        self.graph = KeyframeGraph()
        self.db = DescriptorDatabase()
        self.pending = []           # verified and plausible, waiting for confirmation
        self.loops = {}             # confirmed loops by id
        self.active = set()         # ids of confirmed loops that are currently in the consistent set
        self.rejected = []          # every Rejection so far, for debugging and the demo
        self._pair = {}             # cache of pairwise consistency results
        self._next_id = 0
        self.last_loop = -10 ** 9   # keyframe id of the last accepted loop

    # Old keyframes that look like `kf` and are far enough away along the odometry tree
    def _candidates(self, kf, desc):
        p = self.p
        if len(self.graph) < p.min_keyframes:
            return []
        # Cheap pre-filter on id (ids grow with time), then the exact hop count and path length on what is left
        found = []
        for old_id, score in self.db.query(desc, k=4 * p.top_k, min_score=p.min_score, exclude=lambda j: j > kf.id - p.min_hops):
            if self.graph.hops(old_id, kf.id) < p.min_hops or self.graph.path_length(old_id, kf.id) < p.min_path:
                continue
            found.append((old_id, score))
            if len(found) == p.top_k:
                break
        return found

    def _agree(self, A, B):
        key = (min(A.id, B.id), max(A.id, B.id))
        if key not in self._pair:
            self._pair[key] = consistent(self.graph, A, B, self.p.pcm_chi2)
        return self._pair[key]

    # Process one new keyframe. Returns the changes to the set of loop edges
    def process(self, kf, desc):
        p = self.p
        self.graph.add(kf)
        update = LoopUpdate()

        # 1-3. Retrieve, verify every candidate, gate against odometry
        found = []
        for old_id, score in self._candidates(kf, desc):
            v = self.verifier(old_id, kf.id)
            if v is None or v.n_inliers < p.min_inliers:
                update.rejected.append(Rejection(old_id, kf.id, "verify", 0 if v is None else v.n_inliers))
                continue
            if p.gate:
                d2, ok = plausibility(self.graph, old_id, kf.id, v.T_old_new, v.covariance, p.gate_inflation, p.gate_chi2)
                if not ok:
                    update.rejected.append(Rejection(old_id, kf.id, "gate", d2))
                    continue
            found.append(LoopConstraint(-1, old_id, kf.id, v.T_old_new, v.covariance, v.n_inliers, score, kf.stamp))
        self.db.add(kf.id, desc)
        found.sort(key=lambda L: -L.n_inliers)
        found = found[:p.max_loops_per_kf]

        # 4. Confirmation: a loop is accepted once enough consecutive keyframes found loops that agree with it
        self.pending = [Q for Q in self.pending if kf.id - Q.new_id <= p.confirm_window]
        confirmed = []
        for L in found:
            L.id = self._next_id
            self._next_id += 1
            support = [Q for Q in self.pending
                       if self.graph.hops(L.old_id, Q.old_id) <= p.confirm_hops and consistent(self.graph, L, Q, p.pcm_chi2)]
            if len(support) + 1 >= p.confirm_count:
                # The group is confirmed; keep its best loop (most inliers) and use up the rest
                confirmed.append(max(support + [L], key=lambda Q: Q.n_inliers))
                self.pending = [Q for Q in self.pending if Q not in support]
            else:
                self.pending.append(L)
                update.rejected.append(Rejection(L.old_id, L.new_id, "unconfirmed", L.n_inliers))
        for L in confirmed:
            if kf.id - self.last_loop < p.cooldown:
                update.rejected.append(Rejection(L.old_id, L.new_id, "cooldown", L.n_inliers))
                continue
            self.loops[L.id] = L
            self.last_loop = kf.id
        confirmed = [L for L in confirmed if L.id in self.loops]

        # 5. Keep only the largest mutually consistent group; a new loop can retract an earlier one
        if confirmed:
            self._refresh_active(update)
        self.rejected += update.rejected
        return update

    def _refresh_active(self, update):
        loops = list(self.loops.values())
        if self.p.pcm:
            adj = [[i == j or self._agree(a, b) for j, b in enumerate(loops)] for i, a in enumerate(loops)]
            keep = {loops[i].id for i in best_consistent_set(adj, [L.n_inliers for L in loops])}
        else:
            keep = set(self.loops)
        for L in loops:
            if L.id in keep and L.id not in self.active:
                update.added.append(L)
            elif L.id not in keep and L.id in self.active:
                update.retracted.append(L.id)
            elif L.id not in keep and L.id not in self.active:
                update.rejected.append(Rejection(L.old_id, L.new_id, "inconsistent", L.n_inliers))
        self.active = keep
