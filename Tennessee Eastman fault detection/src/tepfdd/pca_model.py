"""PCA baseline: Hotelling T^2 and SPE (Q) with contribution-plot diagnosis. Fit on fault-free TRAINING runs only."""
import numpy as np
from scipy import stats


class PCAMonitor:
    def __init__(self, cpv=0.90):
        self.cpv = cpv

    def fit(self, X):                      # X: (N, p) fault-free training samples
        self.mu = X.mean(0)
        self.sd = np.maximum(X.std(0, ddof=1), 1e-6)
        Z = (X - self.mu) / self.sd
        cov = np.cov(Z, rowvar=False)
        lam, V = np.linalg.eigh(cov)
        idx = np.argsort(lam)[::-1]
        lam, V = lam[idx], V[:, idx]
        self.lam_all = lam
        self.k = int(np.searchsorted(np.cumsum(lam) / lam.sum(), self.cpv) + 1)
        self.P, self.lam = V[:, :self.k], lam[:self.k]
        self.n = len(X)
        self.W = self.P @ np.diag(1.0 / self.lam) @ self.P.T      # for T^2 and its contributions
        return self

    def stats(self, X):
        Z = (X - self.mu) / self.sd
        t = Z @ self.P
        t2 = (t ** 2 / self.lam).sum(1)
        E = Z - t @ self.P.T
        return t2, (E ** 2).sum(1), Z, E

    def theory_limits(self, alpha=0.99):
        k, n = self.k, self.n
        t2 = k * (n ** 2 - 1) / (n * (n - k)) * stats.f.ppf(alpha, k, n - k)
        res = self.lam_all[self.k:]
        th1, th2, th3 = res.sum(), (res ** 2).sum(), (res ** 3).sum()
        h0 = 1 - 2 * th1 * th3 / (3 * th2 ** 2)
        ca = stats.norm.ppf(alpha)
        spe = th1 * (ca * np.sqrt(2 * th2 * h0 ** 2) / th1 + 1 + th2 * h0 * (h0 - 1) / th1 ** 2) ** (1 / h0)
        return float(t2), float(spe)

    def contribution_features(self, X):
        """104-d normalised contribution vector per sample: [T^2 contributions (clipped >=0), SPE contributions e_j^2]."""
        Z = (X - self.mu) / self.sd
        t = Z @ self.P
        E = Z - t @ self.P.T
        cT = np.clip(Z * (Z @ self.W), 0, None)      # Kourti-MacGregor style: x_j * (P L^-1 P^T x)_j
        cS = E ** 2
        cT /= cT.sum(1, keepdims=True) + 1e-12
        cS /= cS.sum(1, keepdims=True) + 1e-12
        return np.concatenate([cT, cS], 1).astype(np.float32), (t ** 2 / self.lam).sum(1), (E ** 2).sum(1)


class CentroidLibrary:
    """Fault signatures = mean contribution vector over post-onset samples of faulty TRAINING runs; class 0 = normal.
    A sample is diagnosed as the signature with highest cosine similarity (a contribution-plot lookup, not a trained learner)."""
    def __init__(self):
        self.sums, self.cnt = {}, {}

    def add(self, label, F):
        self.sums[label] = self.sums.get(label, 0) + F.sum(0, dtype=np.float64)
        self.cnt[label] = self.cnt.get(label, 0) + len(F)

    def finalize(self):
        self.labels = np.array(sorted(self.sums))
        C = np.stack([self.sums[l] / self.cnt[l] for l in self.labels])
        self.C = C
        self.Cn = (C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)).astype(np.float32)

    def predict(self, F):
        Fn = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-12)
        return self.labels[np.argmax(Fn @ self.Cn.T, axis=1)].astype(np.int8)
