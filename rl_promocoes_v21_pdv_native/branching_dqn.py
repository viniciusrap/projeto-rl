"""Branching DQN — Double DQN com 3 cabeças independentes (Tavakoli 2018).

Mapeamento V20:
  - Cabeça INTENSIDADE (Agente de Desconto):    6 ações
  - Cabeça COMPLEMENTAR (Agente de Combo):      N+1 ações
  - Cabeça ALVO (Agente de Margem):              2 ações

Cada cabeça tem Q-values independentes. Compartilha encoder de estado.
Ações independentes por cabeça reduzem combinatória (252 → 6+21+2 = 29 outputs).

Conceitualmente: 3 "sub-agentes RL" coordenados via observação compartilhada,
treinados por mesmo loss (soma das losses por cabeça). Esta arquitetura é
reconhecida em literatura como forma de MARL coordenado.
"""
from collections import deque
import random
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BranchingDQN(nn.Module):
    """DQN com encoder compartilhado + 3 cabeças (intensidade, complementar, alvo)."""

    def __init__(self, state_dim: int, action_dims: List[int],
                  hidden: int = 128):
        super().__init__()
        self.state_dim = state_dim
        self.action_dims = action_dims  # [6, N+1, 2]
        self.n_heads = len(action_dims)

        # Encoder compartilhado
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

        # Cabeças independentes
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden, 64),
                nn.ReLU(),
                nn.Linear(64, n_act)
            )
            for n_act in action_dims
        ])

    def forward(self, state) -> List[torch.Tensor]:
        """Retorna lista de tensors Q por cabeça [(B, n_act_i)]."""
        h = self.encoder(state)
        return [head(h) for head in self.heads]


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buf.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (
            np.stack(s),
            np.stack(a),
            np.array(r, dtype=np.float32),
            np.stack(ns),
            np.array(d, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buf)


class BranchingDQNAgent:
    """Wrapper que coordena online/target nets + ε-greedy + Double DQN."""

    def __init__(self, state_dim: int, action_dims: List[int],
                  lr: float = 1e-3, gamma: float = 0.95,
                  eps_start: float = 1.0, eps_end: float = 0.05,
                  eps_decay: float = 0.995,
                  buffer_size: int = 50_000, batch_size: int = 64,
                  target_update_steps: int = 500,
                  hidden: int = 128, device: str = 'cpu'):
        self.action_dims = action_dims
        self.gamma = gamma
        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.batch_size = batch_size
        self.target_update_steps = target_update_steps
        self.device = torch.device(device)

        self.online = BranchingDQN(state_dim, action_dims, hidden).to(self.device)
        self.target = BranchingDQN(state_dim, action_dims, hidden).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optim = torch.optim.Adam(self.online.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)
        self.loss_fn = nn.SmoothL1Loss()
        self.steps = 0

    def act(self, state: np.ndarray, greedy: bool = False,
              prior_complementar: np.ndarray = None,
              mask_complementar: np.ndarray = None) -> np.ndarray:
        """Escolhe ação MultiDiscrete por cabeça. ε-greedy se greedy=False.

        prior_complementar: amostragem enviesada por harmonia no random.
        mask_complementar: bool (N+1,) — pares válidos (h>=1.0). Q-values
            inválidos viram -1e9 antes do argmax.
        """
        if not greedy and random.random() < self.eps:
            intens = random.randrange(self.action_dims[0])
            # Random com prior (enviesado por harmonia)
            if prior_complementar is not None and len(prior_complementar) == self.action_dims[1]:
                p = np.asarray(prior_complementar, dtype=np.float64)
                # Aplica máscara também no random (não amostra inválidos)
                if mask_complementar is not None and len(mask_complementar) == self.action_dims[1]:
                    p = p * mask_complementar.astype(np.float64)
                p = np.clip(p, 0.001, None)
                p = p / p.sum()
                comp = int(np.random.choice(self.action_dims[1], p=p))
            elif mask_complementar is not None:
                valid_idx = np.where(mask_complementar)[0]
                comp = int(np.random.choice(valid_idx))
            else:
                comp = random.randrange(self.action_dims[1])
            alvo = random.randrange(self.action_dims[2])
            return np.array([intens, comp, alvo])
        # Greedy
        s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            qs = self.online(s)
        action = []
        for h_idx, q in enumerate(qs):
            q_arr = q.squeeze(0).cpu().numpy().copy() if q.dim() == 2 else q.cpu().numpy().copy()
            if h_idx == 1 and mask_complementar is not None:
                q_arr[~mask_complementar] = -1e9
            action.append(int(np.argmax(q_arr)))
        return np.array(action)

    def remember(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return None
        s, a, r, ns, d = self.buffer.sample(self.batch_size)
        s_t = torch.tensor(s, dtype=torch.float32, device=self.device)
        a_t = torch.tensor(a, dtype=torch.long, device=self.device)        # (B, n_heads)
        r_t = torch.tensor(r, dtype=torch.float32, device=self.device)
        ns_t = torch.tensor(ns, dtype=torch.float32, device=self.device)
        d_t = torch.tensor(d, dtype=torch.float32, device=self.device)

        # Q(s,a) por cabeça
        qs_online = self.online(s_t)
        loss_total = 0.0

        with torch.no_grad():
            # Double DQN: ação argmax do ONLINE no next_state, valor do TARGET
            qs_online_ns = self.online(ns_t)
            qs_target_ns = self.target(ns_t)

        for h_idx in range(len(self.action_dims)):
            # Q(s, a_h) — current
            q_sa = qs_online[h_idx].gather(1, a_t[:, h_idx].unsqueeze(1)).squeeze(1)

            # Double DQN target
            best_next_a = qs_online_ns[h_idx].argmax(dim=1, keepdim=True)
            q_target_next = qs_target_ns[h_idx].gather(1, best_next_a).squeeze(1)
            target = r_t + (1 - d_t) * self.gamma * q_target_next

            loss = self.loss_fn(q_sa, target.detach())
            loss_total = loss_total + loss

        self.optim.zero_grad()
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optim.step()

        self.steps += 1
        if self.steps % self.target_update_steps == 0:
            self.target.load_state_dict(self.online.state_dict())

        return float(loss_total.item())

    def decay_eps(self):
        self.eps = max(self.eps_end, self.eps * self.eps_decay)

    def pretreinar_cabeca_complementar(self, env, n_epochs=100, lr=1e-3):
        """Pré-treina a cabeça COMPLEMENTAR de forma supervisionada usando
        a matriz de harmonia como label.

        Para cada categoria, o Q-value de cada par deve ser proporcional à
        harmonia (escalada). Isso quebra o ótimo local de "complemento
        universal" que aparece no treino end-to-end.
        """
        from torch.optim import Adam
        import torch.nn.functional as F

        # Gera dataset: (estado_canonico, harmonia_targets)
        amostras = []
        n_cats = env.N_CATEGORIAS
        for cat_idx in range(n_cats):
            cat = env.categorias[cat_idx]
            if cat.startswith('cigarro'):
                continue
            # Gera vários estados aleatórios COM essa categoria
            for _ in range(5):
                obs, _ = env.reset()
                env.produto_atual_idx = cat_idx
                obs = env._observar()
                # Target: vetor Q para complementar
                tgt = np.zeros(n_cats + 1, dtype=np.float32)
                tgt[0] = 0.0  # "nenhum" = neutro
                for j in range(n_cats):
                    if j == cat_idx:
                        tgt[j + 1] = -2.0  # mesmo produto = ruim
                    else:
                        h = env._harmonia(cat, env.categorias[j])
                        # Mapa: h 2.5 → +5.0, h 1.5 → +2.0, h 1.0 → 0.0, h 0.5 → -2.5
                        tgt[j + 1] = (h - 1.0) * 5.0
                amostras.append((obs, tgt))

        states = np.stack([a[0] for a in amostras])
        targets = np.stack([a[1] for a in amostras])
        S = torch.tensor(states, dtype=torch.float32, device=self.device)
        T = torch.tensor(targets, dtype=torch.float32, device=self.device)

        # Otimizador só para a cabeça complementar (head[1]) e encoder
        opt = Adam(list(self.online.encoder.parameters())
                    + list(self.online.heads[1].parameters()), lr=lr)

        for epoch in range(n_epochs):
            qs = self.online(S)
            q_comp = qs[1]  # (N_amostras, N_cats+1)
            loss = F.mse_loss(q_comp, T)
            opt.zero_grad()
            loss.backward()
            opt.step()

        # Copia para target net
        self.target.load_state_dict(self.online.state_dict())
        return float(loss.item())

    def save(self, path):
        torch.save({
            'online': self.online.state_dict(),
            'target': self.target.state_dict(),
            'optim': self.optim.state_dict(),
            'eps': self.eps,
            'steps': self.steps,
            'state_dim': self.online.state_dim,
            'action_dims': self.action_dims,
        }, path)

    def load(self, path):
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.online.load_state_dict(ck['online'])
        self.target.load_state_dict(ck['target'])
        try:
            self.optim.load_state_dict(ck['optim'])
        except Exception:
            pass
        self.eps = ck.get('eps', self.eps_end)
        self.steps = ck.get('steps', 0)


if __name__ == '__main__':
    # Smoke test
    state_dim = 83
    action_dims = [6, 21, 2]
    agent = BranchingDQNAgent(state_dim, action_dims, hidden=64)
    s = np.random.randn(state_dim).astype(np.float32)
    a = agent.act(s)
    print(f'Ação MultiDiscrete: {a}')
    # Adiciona transições fake e treina
    for _ in range(100):
        ns = np.random.randn(state_dim).astype(np.float32)
        agent.remember(s, a, np.random.rand(), ns, False)
        s = ns
    loss = agent.train_step()
    print(f'Loss inicial: {loss}')
