"""Place recognition: find old keyframes that look like the current one, without using the drifted position.

Two parts. `Vlad` turns the local descriptors of one image (for us XFeat, 64-d float) into one global vector.
`DescriptorDatabase` stores those vectors and answers "which old keyframes are most similar to this one" with a
brute-force cosine search, which is plenty at keyframe counts of a few thousand.

Why not DBoW2: its stock vocabulary is for binary ORB descriptors and the Python bindings need a source build.
VLAD with a vocabulary trained on our own recordings works with any float descriptor. Anything that produces a
global vector (a learned descriptor such as NetVLAD / DINOv2-based) can be dropped in by feeding the database
directly. Check the licence of weights before depending on them.
"""
import numpy as np


def _normalize_rows(a):
    return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)


# Spherical k-means (cosine similarity) on descriptors pooled from many keyframes: the vocabulary of visual "words"
def _fit_vocabulary(descriptor_sets, k, max_per_image=2048, iters=20, seed=0):
    rng = np.random.default_rng(seed)
    pool = []
    for d in descriptor_sets:
        d = _normalize_rows(np.asarray(d, dtype=np.float64))
        if len(d) > max_per_image:
            d = d[rng.choice(len(d), max_per_image, replace=False)]
        pool.append(d)
    X = np.vstack(pool)
    if len(X) < k:
        raise ValueError(f"need at least {k} descriptors to fit {k} clusters, got {len(X)}")
    C = X[rng.choice(len(X), k, replace=False)]
    for _ in range(iters):
        assign = np.argmax(X @ C.T, axis=1)
        for j in range(k):
            members = X[assign == j]
            if len(members):
                c = members.sum(axis=0)
                C[j] = c / max(np.linalg.norm(c), 1e-12)
    return C


class Vlad:
    def __init__(self, k=64):
        self.k = k
        self.centers = None     # (k, d) unit vectors

    def fit(self, descriptor_sets, max_per_image=2048, iters=20, seed=0):
        self.centers = _fit_vocabulary(descriptor_sets, self.k, max_per_image, iters, seed)

    # One unit-length global vector for one image's descriptors: per-cluster residual sums, each normalized,
    # then signed square root and a final L2 normalization (the usual VLAD recipe)
    def encode(self, descriptors):
        if self.centers is None:
            raise RuntimeError("fit the vocabulary first")
        d = _normalize_rows(np.asarray(descriptors, dtype=np.float64))
        assign = np.argmax(d @ self.centers.T, axis=1)
        v = np.zeros_like(self.centers)
        for j in range(self.k):
            members = d[assign == j]
            if len(members):
                r = (members - self.centers[j]).sum(axis=0)
                v[j] = r / max(np.linalg.norm(r), 1e-12)
        v = v.ravel()
        v = np.sign(v) * np.sqrt(np.abs(v))
        return v / max(np.linalg.norm(v), 1e-12)


# Bag of visual words, the DBoW2 idea without the library: a vocabulary of k words, each image becomes a histogram of
# its words weighted by tf-idf (words that appear in few images count more), L2-normalized. A flat vocabulary instead
# of DBoW2's tree, which is fine at our keyframe counts. Works with any float descriptor (XFeat), no binary-ORB vocabulary
class BagOfWords:
    def __init__(self, k=500):
        self.k = k
        self.centers = None
        self.idf = None

    # Learns the words, and how rare each one is, from the descriptor sets of many keyframes
    def fit(self, descriptor_sets, max_per_image=2048, iters=20, seed=0):
        descriptor_sets = list(descriptor_sets)
        self.centers = _fit_vocabulary(descriptor_sets, self.k, max_per_image, iters, seed)
        df = np.zeros(self.k)
        for d in descriptor_sets:
            words = np.argmax(_normalize_rows(np.asarray(d, dtype=np.float64)) @ self.centers.T, axis=1)
            df[np.unique(words)] += 1
        self.idf = np.log((1 + len(descriptor_sets)) / (1 + df)) + 1.0

    def encode(self, descriptors):
        if self.centers is None:
            raise RuntimeError("fit the vocabulary first")
        d = _normalize_rows(np.asarray(descriptors, dtype=np.float64))
        words = np.argmax(d @ self.centers.T, axis=1)
        v = np.bincount(words, minlength=self.k).astype(float)
        v = np.sqrt(v / max(v.sum(), 1.0)) * self.idf         # square-rooted term frequency, tf-idf weighted
        return v / max(np.linalg.norm(v), 1e-12)


class DescriptorDatabase:
    def __init__(self):
        self.ids = []
        self.vecs = []

    def __len__(self):
        return len(self.ids)

    def add(self, kf_id, vec):
        v = np.asarray(vec, dtype=np.float64).ravel()
        self.ids.append(kf_id)
        self.vecs.append(v / max(np.linalg.norm(v), 1e-12))

    def vector(self, kf_id):
        return self.vecs[self.ids.index(kf_id)] if kf_id in self.ids else None

    # Best matches as [(id, cosine similarity)], most similar first. `exclude(id)` -> True drops a keyframe
    def query(self, vec, k=5, min_score=0.0, exclude=None):
        if not self.ids:
            return []
        v = np.asarray(vec, dtype=np.float64).ravel()
        v = v / max(np.linalg.norm(v), 1e-12)
        scores = np.stack(self.vecs) @ v
        out = []
        for i in np.argsort(-scores):
            if scores[i] < min_score:
                break
            if exclude is not None and exclude(self.ids[i]):
                continue
            out.append((self.ids[i], float(scores[i])))
            if len(out) == k:
                break
        return out


# Re-ranks retrieval results so that places, not single keyframes, compete: a keyframe's score is boosted by the
# scores of its temporal neighbours (ids within `window`) that were also retrieved. Returns one entry per place, the
# best-scoring member, most supported place first. window = 0 leaves the results as they are.
# Reason: a true revisit usually retrieves a run of consecutive keyframes, a fluke match retrieves one
def group_rerank(results, window):
    if window <= 0:
        return list(results)
    score = dict(results)
    support = {i: sum(s for j, s in results if abs(j - i) <= window) for i in score}
    out, used = [], set()
    for i in sorted(score, key=lambda i: -support[i]):
        if i in used:
            continue
        members = [j for j in score if abs(j - i) <= window]
        best = max(members, key=lambda j: score[j])
        out.append((best, score[best]))
        used.update(members)
    return out
