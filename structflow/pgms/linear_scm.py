"""Linear-Gaussian SCM over an arbitrary DAG: exact everything.

Every query in the paper's taxonomy -- conditionals, interventions,
counterfactuals, marginal likelihoods -- is closed-form for a linear-Gaussian
SCM. That makes this the reference implementation against which a learned flow
is scored, and it is why the primary benchmark MDP is linear-Gaussian.

The SCM is  x_v = b_v + sum_{u in pa(v)} W[v,u] x_u + eps_v,  eps_v ~ N(0, q_v).
Stacking, x = b + W x + eps, so with M = (I - W)^{-1}:

    mean = M b,    cov = M diag(q) M^T.
"""

from __future__ import annotations

import numpy as np

__all__ = ["LinearSCM"]

JITTER = 1e-10


class LinearSCM:
    def __init__(self, dag, weights, intercepts=None, noise_var=None, order=None):
        """
        Args:
            dag: {node: [parents]}.
            weights: {node: {parent: coefficient}}.
            intercepts: {node: float}, default 0.
            noise_var: {node: float}, default 1. Deterministic nodes (sums)
                should use 0; a jitter is added for numerical conditioning.
            order: optional explicit variable order; defaults to a topological
                sort of ``dag``.
        """
        self.dag = {k: list(v) for k, v in dag.items()}
        self.order = list(order) if order is not None else _topo(self.dag)
        self.index = {v: i for i, v in enumerate(self.order)}
        d = len(self.order)

        self.W = np.zeros((d, d))
        for v, ps in self.dag.items():
            for p in ps:
                self.W[self.index[v], self.index[p]] = weights[v][p]
        self.b = np.array([(intercepts or {}).get(v, 0.0) for v in self.order])
        self.q = np.array([(noise_var or {}).get(v, 1.0) for v in self.order])

        self._M = np.linalg.inv(np.eye(d) - self.W)
        self.mean = self._M @ self.b
        self.cov = self._M @ np.diag(self.q + JITTER) @ self._M.T

    # ---------------------------------------------------------------- sampling

    def sample(self, n, rng):
        """Draw n joint samples. Returns {node: array of shape (n,)}."""
        eps = rng.normal(size=(n, len(self.order))) * np.sqrt(self.q)
        x = (self._M @ (self.b[None, :] + eps).T).T
        return {v: x[:, self.index[v]] for v in self.order}

    def noise_of(self, sample):
        """Abduction: recover the exogenous noise from a full observation.

        Exact here (it is a triangular solve); for a learned flow it is a
        reverse ODE solve. Accepts and returns {node: array}.
        """
        x = self._stack(sample)
        eps = x - (self.W @ x.T).T - self.b[None, :]
        return {v: eps[:, self.index[v]] for v in self.order}

    def replay(self, noise, interventions=None):
        """Predict: re-run the SCM on given noise, with nodes optionally fixed.

        ``noise_of`` then ``replay`` with an intervention is the three-step
        counterfactual (abduct, act, predict).
        """
        interventions = interventions or {}
        eps = self._stack(noise)
        nsamp = eps.shape[0]
        x = np.zeros((nsamp, len(self.order)))
        for v in self.order:
            i = self.index[v]
            if v in interventions:
                x[:, i] = np.broadcast_to(np.asarray(interventions[v], float), (nsamp,))
                continue
            acc = self.b[i] + eps[:, i]
            for p in self.dag[v]:
                acc = acc + self.W[i, self.index[p]] * x[:, self.index[p]]
            x[:, i] = acc
        return {v: x[:, self.index[v]] for v in self.order}

    # ------------------------------------------------------------ exact queries

    def conditional(self, targets, evidence):
        """Exact p(targets | evidence) as (mean, cov, target_order).

        Args:
            targets: iterable of node names.
            evidence: {node: scalar value}.
        """
        tgt = list(targets)
        ev = list(evidence)
        ti = [self.index[v] for v in tgt]
        ei = [self.index[v] for v in ev]
        if not ei:
            return self.mean[ti], self.cov[np.ix_(ti, ti)], tgt

        Stt = self.cov[np.ix_(ti, ti)]
        Ste = self.cov[np.ix_(ti, ei)]
        See = self.cov[np.ix_(ei, ei)]
        resid = np.array([evidence[v] for v in ev]) - self.mean[ei]
        sol = np.linalg.solve(See, resid)
        mean = self.mean[ti] + Ste @ sol
        cov = Stt - Ste @ np.linalg.solve(See, Ste.T)
        return mean, cov, tgt

    def intervened(self, interventions):
        """A new LinearSCM with the given nodes' incoming edges deleted.

        This is graph surgery: deleting the policy blocks of an MDP yields the
        g-formula. Intervened nodes become deterministic constants.
        """
        dag = {v: ([] if v in interventions else ps) for v, ps in self.dag.items()}
        weights = {v: ({} if v in interventions else
                       {p: self.W[self.index[v], self.index[p]] for p in ps})
                   for v, ps in self.dag.items()}
        intercepts = {v: (float(interventions[v]) if v in interventions
                          else self.b[self.index[v]]) for v in self.order}
        noise = {v: (0.0 if v in interventions else self.q[self.index[v]])
                 for v in self.order}
        return LinearSCM(dag, weights, intercepts, noise, order=self.order)

    def log_prob(self, sample):
        """Exact joint log-density of full observations."""
        x = self._stack(sample)
        d = len(self.order)
        L = np.linalg.cholesky(self.cov)
        z = np.linalg.solve(L, (x - self.mean[None, :]).T)
        return -0.5 * (z ** 2).sum(0) - np.log(np.diag(L)).sum() - 0.5 * d * np.log(2 * np.pi)

    # ------------------------------------------------------------------ helpers

    def _stack(self, d):
        cols = [np.atleast_1d(np.asarray(d[v], float)).ravel() for v in self.order]
        return np.stack(cols, axis=1)


def _topo(dag):
    seen, out = set(), []

    def visit(v):
        if v in seen:
            return
        seen.add(v)
        for p in dag[v]:
            visit(p)
        out.append(v)

    for v in dag:
        visit(v)
    return out
