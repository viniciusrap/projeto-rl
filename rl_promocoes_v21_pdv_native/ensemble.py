"""Ensemble de N seeds. Inferência via Q-mean."""
import numpy as np
import torch
from pathlib import Path

from branching_dqn import BranchingDQNAgent


class EnsembleAgent:
    def __init__(self, state_dim, action_dims, model_paths, hidden=128):
        self.action_dims = action_dims
        self.agents = []
        for p in model_paths:
            a = BranchingDQNAgent(state_dim, action_dims, hidden=hidden, device='cpu')
            a.load(str(p))
            a.online.eval()
            self.agents.append(a)

    def act(self, state, greedy=True, prior_complementar=None):
        """Action = argmax do Q-mean entre os agentes."""
        s = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        all_qs = []  # list of lists of tensors
        with torch.no_grad():
            for a in self.agents:
                qs = a.online(s)
                all_qs.append([q.squeeze(0).numpy() for q in qs])
        # Media
        n_heads = len(self.action_dims)
        q_mean = []
        for h in range(n_heads):
            qm = np.stack([q[h] for q in all_qs]).mean(axis=0)
            q_mean.append(qm)
        action = np.array([int(q.argmax()) for q in q_mean])
        return action, q_mean

    # Métodos pra compatibilidade com act() do BranchingDQNAgent
    @property
    def online(self):
        """Hack: retorna um proxy que faz Q-mean no forward."""
        return self

    def __call__(self, s):
        all_qs = []
        with torch.no_grad():
            for a in self.agents:
                qs = a.online(s)
                all_qs.append(qs)
        # Q-mean por cabeça
        n_heads = len(self.action_dims)
        result = []
        for h in range(n_heads):
            stacked = torch.stack([qs[h] for qs in all_qs])
            result.append(stacked.mean(dim=0))
        return result
