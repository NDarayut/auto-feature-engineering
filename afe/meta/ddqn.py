"""A small, self-contained Double DQN agent (numpy only).

CAFEM (PAKDD 2020) trains a Double DQN to navigate the Feature Transformation
Graph. torch isn't a project dependency (and the online path uses none of this),
so the Q-network is a compact two-layer MLP with hand-written forward/backward
passes -- small enough to be fully inspectable, which matches the plan's intent
that the offline machinery stay simple and auditable.

Double DQN (van Hasselt 2016): the online network selects the greedy next
action, the target network evaluates it, which curbs the Q-value overestimation
plain DQN suffers from. Experience replay + a periodically-synced target
network give the usual stability. Everything is deterministic given ``seed``.
"""

from __future__ import annotations

import random
from collections import deque

import numpy as np


class _MLP:
    """Two-layer tanh MLP: state -> Q-value per action. Manual backprop."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int, rng: np.random.RandomState):
        # He-ish init scaled for tanh.
        self.W1 = rng.randn(in_dim, hidden) * np.sqrt(1.0 / in_dim)
        self.b1 = np.zeros(hidden)
        self.W2 = rng.randn(hidden, out_dim) * np.sqrt(1.0 / hidden)
        self.b2 = np.zeros(out_dim)

    def forward(self, X: np.ndarray):
        z1 = X @ self.W1 + self.b1
        h = np.tanh(z1)
        q = h @ self.W2 + self.b2
        return q, (X, h)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)[0]

    def sgd_step(self, cache, dq: np.ndarray, lr: float):
        """Backprop a loss gradient ``dq`` (dL/dQ) and apply an SGD update."""
        X, h = cache
        n = X.shape[0]
        gW2 = h.T @ dq / n
        gb2 = dq.mean(axis=0)
        dh = (dq @ self.W2.T) * (1.0 - h ** 2)
        gW1 = X.T @ dh / n
        gb1 = dh.mean(axis=0)
        for p, g in ((self.W1, gW1), (self.b1, gb1),
                     (self.W2, gW2), (self.b2, gb2)):
            np.clip(g, -1.0, 1.0, out=g)
            p -= lr * g

    def copy_from(self, other: "_MLP"):
        self.W1, self.b1 = other.W1.copy(), other.b1.copy()
        self.W2, self.b2 = other.W2.copy(), other.b2.copy()


class DoubleDQN:
    """Double DQN with experience replay over a discrete action space."""

    def __init__(
        self, state_dim: int, n_actions: int, hidden: int = 64,
        gamma: float = 0.9, lr: float = 5e-3, batch_size: int = 32,
        buffer_size: int = 5000, target_sync: int = 100,
        eps_start: float = 1.0, eps_end: float = 0.1, eps_decay: float = 0.98,
        seed: int = 0,
    ):
        self.n_actions = n_actions
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.target_sync = target_sync
        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay

        self._rng = np.random.RandomState(seed)
        self._py_rng = random.Random(seed)
        self.online = _MLP(state_dim, hidden, n_actions, self._rng)
        self.target = _MLP(state_dim, hidden, n_actions, self._rng)
        self.target.copy_from(self.online)
        self.replay: deque = deque(maxlen=buffer_size)
        self._learn_steps = 0

    def act(self, state: np.ndarray, greedy: bool = False) -> int:
        """Epsilon-greedy action selection (greedy at eval time)."""
        if not greedy and self._py_rng.random() < self.eps:
            return self._py_rng.randrange(self.n_actions)
        q = self.online.predict(state[None, :])[0]
        return int(np.argmax(q))

    def remember(self, s, a, r, s_next, done):
        self.replay.append((s, a, r, s_next, done))

    def _sample(self):
        batch = self._py_rng.sample(self.replay, self.batch_size)
        s = np.array([b[0] for b in batch], dtype="float64")
        a = np.array([b[1] for b in batch], dtype="int64")
        r = np.array([b[2] for b in batch], dtype="float64")
        s_next = np.array([b[3] for b in batch], dtype="float64")
        done = np.array([b[4] for b in batch], dtype="float64")
        return s, a, r, s_next, done

    def learn(self) -> float | None:
        """One minibatch Double-DQN update. Returns the batch MSE (or None)."""
        if len(self.replay) < self.batch_size:
            return None
        s, a, r, s_next, done = self._sample()

        # Double DQN target: online picks the argmax action, target scores it.
        next_online = self.online.predict(s_next)
        next_actions = np.argmax(next_online, axis=1)
        next_target = self.target.predict(s_next)
        next_q = next_target[np.arange(len(a)), next_actions]
        td_target = r + self.gamma * next_q * (1.0 - done)

        q, cache = self.online.forward(s)
        q_taken = q[np.arange(len(a)), a]
        err = q_taken - td_target  # dL/dQ_taken for 0.5*MSE

        dq = np.zeros_like(q)
        dq[np.arange(len(a)), a] = err
        self.online.sgd_step(cache, dq, self.lr)

        self._learn_steps += 1
        if self._learn_steps % self.target_sync == 0:
            self.target.copy_from(self.online)
        return float(np.mean(err ** 2))

    def decay_epsilon(self):
        self.eps = max(self.eps_end, self.eps * self.eps_decay)
