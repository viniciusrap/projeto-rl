"""Modelos DQN para V12 — BranchingDQN + ReplayBuffer.

BranchingDQN: arquitetura dueling com cabeças decompostas para ação
MultiDiscrete([N+1, 5]). Reduz espaço de saída de N×5 para N+5 outputs.

Q(s,(p,i)) = V(s) + A_prod(s,p) + A_int(s,i)

ReplayBuffer: deque circular para experience replay.
"""
from collections import deque

import numpy as np
import torch
import torch.nn as nn


class BranchingDQN(nn.Module):
    """DQN com cabeças decompostas (dueling-style) para ação MultiDiscrete."""

    def __init__(self, obs_dim: int, n_produtos: int, n_intensidades: int):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
        )
        self.value = nn.Linear(128, 1)
        self.adv_prod = nn.Sequential(nn.Linear(128, 64), nn.ReLU(),
                                        nn.Linear(64, n_produtos))
        self.adv_int = nn.Sequential(nn.Linear(128, 64), nn.ReLU(),
                                       nn.Linear(64, n_intensidades))
        self.n_produtos = n_produtos
        self.n_intensidades = n_intensidades

    def forward(self, x):
        h = self.trunk(x)
        v = self.value(h)
        ap = self.adv_prod(h) - self.adv_prod(h).mean(dim=-1, keepdim=True)
        ai = self.adv_int(h) - self.adv_int(h).mean(dim=-1, keepdim=True)
        return v, ap, ai

    def q_values(self, x):
        """Q(s, p, i) tensor de shape (B, N_p, N_i)."""
        v, ap, ai = self.forward(x)
        return v.unsqueeze(-1) + ap.unsqueeze(-1) + ai.unsqueeze(-2)

    def select_action(self, x, eps: float, rng):
        """ε-greedy: retorna (produto, intensidade)."""
        if rng.random() < eps:
            return (int(rng.integers(0, self.n_produtos)),
                    int(rng.integers(0, self.n_intensidades)))
        with torch.no_grad():
            q = self.q_values(x)
            flat = q.view(-1)
            idx = int(flat.argmax().item())
            p = idx // self.n_intensidades
            i = idx % self.n_intensidades
            return p, i


class ReplayBuffer:
    """Experience replay buffer (circular deque)."""

    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)

    def push(self, s, ap, ai, r, s2, done):
        self.buf.append((s, ap, ai, r, s2, done))

    def sample(self, batch_size: int, rng):
        idxs = rng.integers(0, len(self.buf), size=batch_size)
        batch = [self.buf[i] for i in idxs]
        s = np.array([b[0] for b in batch], dtype=np.float32)
        ap = np.array([b[1] for b in batch], dtype=np.int64)
        ai = np.array([b[2] for b in batch], dtype=np.int64)
        r = np.array([b[3] for b in batch], dtype=np.float32)
        s2 = np.array([b[4] for b in batch], dtype=np.float32)
        d = np.array([b[5] for b in batch], dtype=np.float32)
        return s, ap, ai, r, s2, d

    def __len__(self):
        return len(self.buf)
