"""Incremental pose graph backend (GTSAM iSAM2).

The batch script (prototype/pose_graph.py) solves the whole graph once at the end. This keeps one running solution:
odometry edges are added as keyframes arrive, and loop edges come from the loop closure detector's LoopUpdate. iSAM2
only re-solves the part of the graph a new edge touches, so it keeps up live.

Loops are retractable. Every loop edge's factor index is remembered, so a loop the detector takes back (`retracted`)
is removed from the graph, the same way RTAB-Map does it. After a loop goes in, its residual is checked against its
own covariance: if the rest of the graph strongly disagrees (the RTAB-Map "optimization error ratio" idea), the loop
is removed again and reported back as rejected. Loop edges get a Huber loss as a further safety net.

Needs gtsam. ROS-free.
"""
import gtsam
import numpy as np
from gtsam.symbol_shorthand import X

from slam_core.loop_closure.gate import CHI2_6_999
from slam_core.loop_closure.se3 import mahalanobis2, pose_error


class IncrementalBackend:
    def __init__(self, huber=1.345, max_loop_chi2=CHI2_6_999, relinearize_threshold=0.05):
        params = gtsam.ISAM2Params()
        params.setRelinearizeThreshold(relinearize_threshold)
        params.relinearizeSkip = 1
        self.isam = gtsam.ISAM2(params)
        self.huber = gtsam.noiseModel.mEstimator.Huber.Create(huber)
        self.max_loop_chi2 = max_loop_chi2
        self.estimate = gtsam.Values()
        self.loops = {}             # loop id -> (factor index, LoopConstraint)

    # Pose of keyframe i in the odom frame (4x4)
    def pose(self, i):
        return self.estimate.atPose3(X(i)).matrix()

    def poses(self):
        return {i: self.pose(i) for i in self.keys()}

    def keys(self):
        return sorted(gtsam.Symbol(k).index() for k in self.estimate.keys())

    # Adds one keyframe: a prior if it is the first, else an odometry edge to its previous keyframe
    def add_keyframe(self, kf):
        graph, values = gtsam.NonlinearFactorGraph(), gtsam.Values()
        if kf.is_first:
            tight = gtsam.noiseModel.Diagonal.Sigmas(np.full(6, 1e-6))
            graph.add(gtsam.PriorFactorPose3(X(kf.id), gtsam.Pose3(kf.T_odom_kf), tight))
            values.insert(X(kf.id), gtsam.Pose3(kf.T_odom_kf))
        else:
            noise = gtsam.noiseModel.Gaussian.Covariance(kf.covariance)
            graph.add(gtsam.BetweenFactorPose3(X(kf.prev_id), X(kf.id), gtsam.Pose3(kf.T_prev_kf), noise))
            values.insert(X(kf.id), self.estimate.atPose3(X(kf.prev_id)).compose(gtsam.Pose3(kf.T_prev_kf)))
        self.isam.update(graph, values)
        self.estimate = self.isam.calculateEstimate()

    # Applies a LoopUpdate from the detector. Returns the ids of loops this backend rejected after optimizing
    def apply(self, update):
        graph = gtsam.NonlinearFactorGraph()
        for L in update.added:
            noise = gtsam.noiseModel.Robust.Create(self.huber, gtsam.noiseModel.Gaussian.Covariance(L.covariance))
            graph.add(gtsam.BetweenFactorPose3(X(L.old_id), X(L.new_id), gtsam.Pose3(L.T_old_new), noise))
        remove = [self.loops.pop(i)[0] for i in update.retracted if i in self.loops]
        if not update.added and not remove:
            return []
        res = self.isam.update(graph, gtsam.Values(), remove)
        for L, idx in zip(update.added, res.getNewFactorsIndices()):
            self.loops[L.id] = (idx, L)
        self.estimate = self.isam.calculateEstimate()

        # The graph-level check: a loop the rest of the graph cannot live with is taken back out
        bad = [L.id for L in update.added if self._loop_chi2(L) > self.max_loop_chi2]
        if bad:
            self.isam.update(gtsam.NonlinearFactorGraph(), gtsam.Values(), [self.loops.pop(i)[0] for i in bad])
            self.estimate = self.isam.calculateEstimate()
        return bad

    # Squared Mahalanobis residual of loop L under the current solution
    def _loop_chi2(self, L):
        T_est = np.linalg.inv(self.pose(L.old_id)) @ self.pose(L.new_id)
        return mahalanobis2(pose_error(T_est, L.T_old_new), L.covariance)
