"""Modelos DQN para V12 — BranchingDQN + ReplayBuffer + CondPolicyDQN.

BranchingDQN: arquitetura dueling com cabeças decompostas para ação
MultiDiscrete([N+1, 5]). Reduz espaço de saída de N×5 para N+5 outputs.

Q(s,(p,i)) = V(s) + A_prod(s,p) + A_int(s,i)

V12.3 — extensões:
- BranchingDQN.select_action() aceita `mask_cat` opcional (hard mask em Q-values
  via torch.finfo(dtype).min). Usado para proibir cigarros (Lei 9.294/96)
  categoricamente em vez de penalty leve.
- CondPolicyDQN: decomposição condicional π_cat(s) → π_int(s, cat_chosen) via
  embedding. Quebra independência aditiva do BranchingDQN — permite que
  liquid25%/desc10% emerjam condicionados ao produto certo (vencimento alto).
  Inspirado em AlphaStar (DeepMind 2019) e Hierarchical DQN (Kulkarni 2016).

ReplayBuffer: deque circular para experience replay.
"""
from collections import deque

import numpy as np
import torch
import torch.nn as nn

# Sentinel para hard mask em Q-values (seguro numericamente)
NEG_INF = torch.finfo(torch.float32).min


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

    def select_action(self, x, eps: float, rng, mask_cat=None):
        """ε-greedy: retorna (produto, intensidade).

        V12.3: se `mask_cat` (np.bool array shape [n_produtos]) for fornecido,
        aplica hard mask via NEG_INF nos Q-values das categorias inválidas.
        Garante prob=0 de escolher ação proibida (cigarro, etc.).

        Args:
            mask_cat: array booleano shape (n_produtos,). True = válido. None = sem mask.
        """
        # Constrói lista de produtos válidos (para sampleamento ε)
        if mask_cat is not None:
            valid_prods = np.where(mask_cat)[0]
            if len(valid_prods) == 0:
                valid_prods = np.array([0])  # fallback: sem-promo sempre OK
        else:
            valid_prods = np.arange(self.n_produtos)

        if rng.random() < eps:
            return (int(rng.choice(valid_prods)),
                    int(rng.integers(0, self.n_intensidades)))
        with torch.no_grad():
            q = self.q_values(x)  # (1, N_p, N_i)
            if mask_cat is not None:
                # Mask categoria: linha inteira (todas intensidades) vira NEG_INF
                mask_t = torch.tensor(mask_cat, dtype=torch.bool,
                                       device=q.device).unsqueeze(0).unsqueeze(-1)
                q = q.masked_fill(~mask_t, NEG_INF)
            flat = q.view(-1)
            idx = int(flat.argmax().item())
            p = idx // self.n_intensidades
            i = idx % self.n_intensidades
            return p, i


class BranchingDQNCross(BranchingDQN):
    """V13.v2 — BranchingDQN + cross-head para cruzar produto×intensidade.

    Híbrido: mantém arquitetura V12.1 (que aprendeu eventos) e adiciona
    pequena cabeça de cruzamento explícito p×i para liberar liq25%/desc10%.

    Q(s, p, i) = V(s) + A_prod(s, p) + A_int(s, i) + Cross(s, p, i)

    Cross-head tem n_produtos × n_intensidades outputs centralizados.
    """

    def __init__(self, obs_dim: int, n_produtos: int, n_intensidades: int,
                 cross_hidden: int = 64):
        super().__init__(obs_dim, n_produtos, n_intensidades)
        self.cross = nn.Sequential(
            nn.Linear(128, cross_hidden), nn.ReLU(),
            nn.Linear(cross_hidden, n_produtos * n_intensidades),
        )

    def q_values(self, x):
        h = self.trunk(x)
        v = self.value(h)
        ap_raw = self.adv_prod(h)
        ai_raw = self.adv_int(h)
        ap = ap_raw - ap_raw.mean(dim=-1, keepdim=True)
        ai = ai_raw - ai_raw.mean(dim=-1, keepdim=True)
        cross = self.cross(h).view(-1, self.n_produtos, self.n_intensidades)
        # Centraliza p e i: subtrai a média global em (p, i)
        cross = cross - cross.mean(dim=(-1, -2), keepdim=True)
        return v.unsqueeze(-1) + ap.unsqueeze(-1) + ai.unsqueeze(-2) + cross


class CondPolicyDQN(nn.Module):
    """V12.3 — Decomposição condicional para destravar diversidade de ações.

    Política autoregressiva: primeira cabeça escolhe categoria, segunda cabeça
    recebe embedding da categoria escolhida + estado e decide intensidade.
    Quebra independência aditiva do BranchingDQN — permite que liquid25%
    emerja condicionado a produto perto de vencer (alta idade/validade).

    Q_cat(s) = MLP(s)
    Q_int(s, cat) = MLP(concat[s, emb(cat)])

    Cases: AlphaStar (sequencial sobre ação composta), Hierarchical DQN
    Kulkarni 2016 (Montezuma's Revenge).
    """

    def __init__(self, obs_dim: int, n_produtos: int, n_intensidades: int,
                 emb_dim: int = 16, hidden: int = 128):
        super().__init__()
        self.n_produtos = n_produtos
        self.n_intensidades = n_intensidades
        self.emb_dim = emb_dim

        self.enc = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, hidden), nn.ReLU(),
        )
        self.q_cat_head = nn.Linear(hidden, n_produtos)
        self.cat_emb = nn.Embedding(n_produtos, emb_dim)
        self.q_int_head = nn.Sequential(
            nn.Linear(hidden + emb_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, n_intensidades),
        )

    def forward(self, x, cat=None):
        """Forward que retorna (Q_cat, Q_int).

        Se cat=None: usa argmax(Q_cat) para condicionar Q_int.
        Se cat fornecido: condiciona Q_int em cat dado (para batch training).
        """
        z = self.enc(x)
        q_cat = self.q_cat_head(z)
        if cat is None:
            cat = q_cat.argmax(dim=-1)
        cat = cat.long()
        e = self.cat_emb(cat)
        zi = torch.cat([z, e], dim=-1)
        q_int = self.q_int_head(zi)
        return q_cat, q_int, cat

    def select_action(self, x, eps: float, rng, mask_cat=None):
        """ε-greedy: (cat, int). Suporta hard mask de categoria."""
        if mask_cat is not None:
            valid_prods = np.where(mask_cat)[0]
            if len(valid_prods) == 0:
                valid_prods = np.array([0])
        else:
            valid_prods = np.arange(self.n_produtos)

        if rng.random() < eps:
            cat = int(rng.choice(valid_prods))
            inten = int(rng.integers(0, self.n_intensidades))
            return cat, inten

        with torch.no_grad():
            z = self.enc(x)
            q_cat = self.q_cat_head(z)
            if mask_cat is not None:
                mask_t = torch.tensor(mask_cat, dtype=torch.bool,
                                       device=q_cat.device).unsqueeze(0)
                q_cat = q_cat.masked_fill(~mask_t, NEG_INF)
            cat = int(q_cat.argmax(dim=-1).item())
            e = self.cat_emb(torch.tensor([cat], device=z.device))
            zi = torch.cat([z, e], dim=-1)
            q_int = self.q_int_head(zi)
            inten = int(q_int.argmax(dim=-1).item())
        return cat, inten


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


class PrioritizedReplayBuffer:
    """V14 — Prioritized Experience Replay (Schaul et al. 2016).

    Sample probabilidade ~ |TD-error|^alpha. Transições raras (eventos
    comerciais) são amostradas mais → modelo aprende essas datas melhor.

    Importance-Sampling weights corrigem bias: w_i = (N * p_i)^(-beta).
    Beta cresce ~0→1 ao longo do treino (annealing).
    """

    def __init__(self, capacity: int, alpha: float = 0.6, beta_start: float = 0.4,
                 beta_increment: float = 1e-4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta_start
        self.beta_increment = beta_increment
        self.buf = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos = 0

    def push(self, s, ap, ai, r, s2, done):
        max_prio = self.priorities[:len(self.buf)].max() if self.buf else 1.0
        if len(self.buf) < self.capacity:
            self.buf.append((s, ap, ai, r, s2, done))
        else:
            self.buf[self.pos] = (s, ap, ai, r, s2, done)
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int, rng):
        n = len(self.buf)
        prios = self.priorities[:n]
        probs = prios ** self.alpha
        probs_sum = probs.sum()
        if probs_sum <= 0:
            probs = np.ones(n) / n
        else:
            probs = probs / probs_sum

        indices = rng.choice(n, batch_size, p=probs)
        batch = [self.buf[i] for i in indices]
        s = np.array([b[0] for b in batch], dtype=np.float32)
        ap = np.array([b[1] for b in batch], dtype=np.int64)
        ai = np.array([b[2] for b in batch], dtype=np.int64)
        r = np.array([b[3] for b in batch], dtype=np.float32)
        s2 = np.array([b[4] for b in batch], dtype=np.float32)
        d = np.array([b[5] for b in batch], dtype=np.float32)

        # Importance-Sampling weights
        weights = (n * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()  # normaliza
        weights = weights.astype(np.float32)

        # Anneal beta toward 1
        self.beta = min(1.0, self.beta + self.beta_increment)

        return s, ap, ai, r, s2, d, indices, weights

    def update_priorities(self, indices, td_errors):
        """Atualiza prioridades com |TD-error| + ε."""
        for i, td in zip(indices, td_errors):
            self.priorities[i] = float(abs(td)) + 1e-6

    def __len__(self):
        return len(self.buf)
