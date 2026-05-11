"""V10: adiciona features explícitas de 'período fraco' ao estado.

Combina com V9 (reward shaping) para criar mapeamento trivial:
- Estado contém: fraco_flag[P] (binário, 1 se contexto atual é fraco para P)
- Recompensa: bonus por promover em fraco, penalidade por promover em forte
- Política esperada: agente aprende a promover quando fraco_flag indica
"""
import json

NB = r'C:\Users\vinin\projeto-rl\notebooks\rl_conveniencia_viana_FINAL.ipynb'
nb = json.load(open(NB, encoding='utf-8'))

# 1. Modificar env _get_obs para incluir fraco_flags
for i, c in enumerate(nb['cells']):
    if c['cell_type'] == 'code' and 'class ConvenienceStoreEnv' in ''.join(c.get('source', '')):
        src = ''.join(c['source'])

        # Mudar observation_space de 41 para 47
        src = src.replace(
            'self.observation_space = spaces.Box(0., 1., shape=(41,), dtype=np.float32)',
            'self.observation_space = spaces.Box(0., 1., shape=(47,), dtype=np.float32)'
        )

        # Modificar _get_obs para adicionar 6 features de fraco_flag
        old_obs = '''    def _get_obs(self):
        t = self._step % 3
        d = (self._step // 3) % 7
        m = self._mes
        to = np.zeros(3); to[t] = 1.
        do = np.zeros(7); do[d] = 1.
        mo = np.zeros(12); mo[m] = 1.
        # Estado: nível de estoque (referência) + validade restante (1 - risco)
        estoque_norm  = np.clip(self.estoque / (ESTOQUE_INICIAL * 2.0), 0, 1)
        validade_rest = np.clip(1 - self.idade / VALIDADE_TIPICA, 0, 1)
        return np.concatenate([
            to, do, mo, [self._temp_norm],
            estoque_norm, validade_rest, self.promo_ant
        ]).astype(np.float32)'''
        new_obs = '''    def _get_obs(self):
        t = self._step % 3
        d = (self._step // 3) % 7
        m = self._mes
        to = np.zeros(3); to[t] = 1.
        do = np.zeros(7); do[d] = 1.
        mo = np.zeros(12); mo[m] = 1.
        # Estado: nível de estoque (referência) + validade restante (1 - risco)
        estoque_norm  = np.clip(self.estoque / (ESTOQUE_INICIAL * 2.0), 0, 1)
        validade_rest = np.clip(1 - self.idade / VALIDADE_TIPICA, 0, 1)
        # V10: fraco_flag por produto — sinal explícito de "este produto está em
        # contexto historicamente fraco" (bottom 30% do fator combinado)
        fraco_flag = np.zeros(N_PRODUTOS, dtype=np.float32)
        for _p in range(N_PRODUTOS):
            _fator_now = FATOR_DIA[_p, d] * FATOR_TURNO[_p, t] * FATOR_MES[_p, m]
            if _fator_now < LIMIAR_FRACO_PROD[_p]:
                fraco_flag[_p] = 1.
        return np.concatenate([
            to, do, mo, [self._temp_norm],
            estoque_norm, validade_rest, self.promo_ant, fraco_flag
        ]).astype(np.float32)'''
        src = src.replace(old_obs, new_obs)

        # Atualizar docstring V9 → V10
        src = src.replace('"""Ambiente V9 ', '"""Ambiente V10 ')
        src = src.replace("'✓ Ambiente V9 validado", "'✓ Ambiente V10 validado")
        # Atualizar assert do shape
        src = src.replace('assert obs.shape == (41,)', 'assert obs.shape == (47,)')

        c['source'] = src.splitlines(keepends=True)
        c['outputs'] = []
        c['execution_count'] = None
        print(f'Cell {i}: env V10 com 6 fraco_flags (estado 47 features)')
        break

# 2. Atualizar arquitetura DQN para 47 inputs
for i, c in enumerate(nb['cells']):
    if c['cell_type'] == 'code' and 'class DQN(nn.Module)' in ''.join(c.get('source', '')):
        src = ''.join(c['source'])
        src = src.replace('def __init__(self,inp=41', 'def __init__(self,inp=47')
        src = src.replace('Linear(41,128)', 'Linear(47,128)')
        c['source'] = src.splitlines(keepends=True)
        c['outputs'] = []
        c['execution_count'] = None
        print(f'Cell {i}: DQN input expandido para 47 features')
        break

# 3. Atualizar get_recommendation para usar shape 47
for i, c in enumerate(nb['cells']):
    if c['cell_type'] == 'code' and 'def get_recommendation' in ''.join(c.get('source', '')):
        src = ''.join(c['source'])
        # Just rebuild via env._get_obs() — função já usa env, então shape ajusta automaticamente
        c['source'] = src.splitlines(keepends=True)
        c['outputs'] = []
        c['execution_count'] = None
        print(f'Cell {i}: get_recommendation (sem mudança — usa env._get_obs)')
        break

# 4. Atualizar query_agent na validação 7.4 para construir estado de 47 features
for i, c in enumerate(nb['cells']):
    if c['cell_type'] == 'code' and 'def query_agent' in ''.join(c.get('source', '')):
        src = ''.join(c['source'])
        old_query = '''def query_agent(dia, turno, mes, prod_critico):
    """Estado canônico com prod_critico em risco alto de vencimento."""
    to = np.zeros(3); to[turno] = 1
    do = np.zeros(7); do[dia] = 1
    mo = np.zeros(12); mo[mes] = 1
    temp = float(TEMP_MEDIA_MES_NORM[mes])
    estoque_norm = np.ones(N_PRODUTOS) * 0.7
    # Produto crítico: validade muito baixa (idade alta)
    validade_rest = np.ones(N_PRODUTOS) * 0.6
    validade_rest[prod_critico] = 0.15  # 85% da validade já usada
    promo_ant = np.zeros(N_PRODUTOS)
    obs = np.concatenate([to, do, mo, [temp], estoque_norm, validade_rest, promo_ant]).astype(np.float32)
    with torch.no_grad():
        q = agent_final.q(torch.tensor(obs).unsqueeze(0).to(device)).squeeze().cpu().numpy()
    return int(q.argmax())'''
        new_query = '''def query_agent(dia, turno, mes, prod_critico):
    """Estado canônico com prod_critico em risco alto de vencimento (V10: 47 features)."""
    to = np.zeros(3); to[turno] = 1
    do = np.zeros(7); do[dia] = 1
    mo = np.zeros(12); mo[mes] = 1
    temp = float(TEMP_MEDIA_MES_NORM[mes])
    estoque_norm = np.ones(N_PRODUTOS) * 0.7
    validade_rest = np.ones(N_PRODUTOS) * 0.6
    validade_rest[prod_critico] = 0.15
    promo_ant = np.zeros(N_PRODUTOS)
    # V10: fraco_flag por produto
    fraco_flag = np.zeros(N_PRODUTOS, dtype=np.float32)
    for _p in range(N_PRODUTOS):
        _fator_now = FATOR_DIA[_p, dia] * FATOR_TURNO[_p, turno] * FATOR_MES[_p, mes]
        if _fator_now < LIMIAR_FRACO_PROD[_p]:
            fraco_flag[_p] = 1.
    obs = np.concatenate([to, do, mo, [temp], estoque_norm, validade_rest, promo_ant, fraco_flag]).astype(np.float32)
    with torch.no_grad():
        q = agent_final.q(torch.tensor(obs).unsqueeze(0).to(device)).squeeze().cpu().numpy()
    return int(q.argmax())'''
        if old_query in src:
            src = src.replace(old_query, new_query)
            print(f'Cell {i}: query_agent atualizada para 47 features')
        c['source'] = src.splitlines(keepends=True)
        c['outputs'] = []
        c['execution_count'] = None
        break

with open(NB, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print('V10 patches aplicados.')
