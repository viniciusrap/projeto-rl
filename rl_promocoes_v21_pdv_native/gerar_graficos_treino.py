"""Gera gráficos de curva de aprendizado ESTÁVEIS a partir dos logs de
treino multi-seed.

Cada seed produz um CSV com 1 linha por episódio. Este script:
- Lê todos os CSVs de uma iteração (todas as seeds)
- Calcula média e desvio-padrão entre seeds por episódio
- Plota com banda de confiança (média ± 1 desvio) — padrão acadêmico
- Aplica média móvel para suavizar (curva estável visualmente)

Saída: results/v21/curvas_treino.png  (4 painéis)
"""
import io
import sys
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def media_movel(x, janela=30):
    if len(x) < janela:
        return x
    return np.convolve(x, np.ones(janela) / janela, mode='valid')


def gerar(iter_nome, log_dir, out_png):
    arquivos = sorted(glob.glob(str(log_dir / f'iter_{iter_nome}_seed_*.csv')))
    if not arquivos:
        print(f'❌ Nenhum log encontrado para iter_{iter_nome}_seed_*')
        return False

    print(f'Lendo {len(arquivos)} seeds:')
    dfs = []
    for a in arquivos:
        df = pd.read_csv(a)
        dfs.append(df)
        print(f'  {Path(a).name}: {len(df)} episódios')

    # Alinha pelo menor número de episódios
    min_ep = min(len(d) for d in dfs)
    dfs = [d.iloc[:min_ep].reset_index(drop=True) for d in dfs]
    n_seeds = len(dfs)

    # Empilha métricas: (n_seeds, n_episodios)
    def stack(col):
        return np.stack([d[col].values for d in dfs])

    eps = np.arange(min_ep)
    jan = 30
    # 1 seed → banda = volatilidade LOCAL (desvio móvel da própria curva).
    # N seeds → banda = desvio ENTRE seeds.
    um_seed = n_seeds == 1
    banda_lbl = ('± volatilidade local (janela 50)' if um_seed
                  else '± 1 desvio entre seeds')

    def rolling_std(x, w=50):
        s = pd.Series(x)
        return s.rolling(w, min_periods=1).std().fillna(0).values

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    modo = (f'1 seed × {min_ep} episódios (convergência longa)' if um_seed
             else f'{n_seeds} seeds × {min_ep} episódios')
    fig.suptitle(f'Curvas de Aprendizado — V21 PDV-native ({modo})',
                  fontsize=15, fontweight='bold')

    # ── Painel 1: Recompensa ──
    ax = axes[0, 0]
    R = stack('reward')
    media = R.mean(axis=0)
    jan = 30
    media_s = media_movel(media, jan)
    eps_s = eps[:len(media_s)]
    if um_seed:
        banda = rolling_std(media, 50)[:len(media_s)]
    else:
        banda = media_movel(R.std(axis=0), jan)
    ax.plot(eps_s, media_s, color='#2563eb', lw=2, label='Recompensa (suavizada)')
    ax.fill_between(eps_s, media_s - banda, media_s + banda,
                     color='#2563eb', alpha=0.18, label=banda_lbl)
    ax.axhline(0, color='#94a3b8', ls='--', lw=1)
    ax.set_title('Recompensa por episódio (suavizada, janela 30)', fontweight='bold')
    ax.set_xlabel('Episódio'); ax.set_ylabel('Recompensa')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(alpha=0.25)

    # ── Painel 2: Recompensa média móvel 30ep (já no log) ──
    ax = axes[0, 1]
    A = stack('avg_reward_30ep')
    m = A.mean(axis=0)
    if um_seed:
        banda2 = rolling_std(m, 50)
    else:
        banda2 = A.std(axis=0)
    ax.plot(eps, m, color='#16a34a', lw=2, label='Média móvel 30ep')
    ax.fill_between(eps, m - banda2, m + banda2, color='#16a34a', alpha=0.18,
                     label=banda_lbl)
    ax.set_title('Recompensa média móvel (convergência)', fontweight='bold')
    ax.set_xlabel('Episódio'); ax.set_ylabel('Recompensa média 30ep')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(alpha=0.25)

    # ── Painel 3: Loss (aprendizado da rede) ──
    ax = axes[1, 0]
    if 'loss' in dfs[0].columns:
        L = stack('loss')
        m = L.mean(axis=0)
        m_s = media_movel(m, jan)
        ax.plot(eps[:len(m_s)], m_s, color='#dc2626', lw=2)
        ax.set_title('Loss da rede (suavizada)', fontweight='bold')
        ax.set_xlabel('Episódio'); ax.set_ylabel('Loss (Huber)')
        ax.grid(alpha=0.25)

    # ── Painel 4: % combo vs nada (política aprendida) ──
    ax = axes[1, 1]
    if 'combo_pct' in dfs[0].columns:
        C = stack('combo_pct')
        mc = media_movel(C.mean(axis=0), jan)
        ax.plot(eps[:len(mc)], mc * 100, color='#9333ea', lw=2,
                 label='% Combo')
    if 'promoveu_pct' in dfs[0].columns:
        P = stack('promoveu_pct')
        mp = media_movel(P.mean(axis=0), jan)
        ax.plot(eps[:len(mp)], mp * 100, color='#f59e0b', lw=2,
                 label='% Promoveu (qualquer)')
    ax.set_title('Política aprendida (% das ações)', fontweight='bold')
    ax.set_xlabel('Episódio'); ax.set_ylabel('%')
    ax.set_ylim(0, 100)
    ax.legend(loc='best', fontsize=9)
    ax.grid(alpha=0.25)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=130, bbox_inches='tight')
    plt.close()

    # Estatísticas finais (texto)
    print(f'\n{"="*60}')
    if um_seed:
        # 1 seed: estabilidade = quão lisa ficou a curva no fim
        # Compara volatilidade do início vs fim do treino
        curva = R[0]
        ini = curva[100:600]      # episódios 100-600 (após aquecer)
        fim = curva[-500:]         # últimos 500
        cv_ini = ini.std() / abs(ini.mean()) if ini.mean() else 0
        cv_fim = fim.std() / abs(fim.mean()) if fim.mean() else 0
        print(f'ESTABILIDADE (1 seed × {min_ep} ep — convergência da curva):')
        print(f'  Recompensa média (últimos 500 ep): {fim.mean():>10.1f}')
        print(f'  Volatilidade no início (ep 100-600): CV={cv_ini:.3f}')
        print(f'  Volatilidade no fim (últimos 500)  : CV={cv_fim:.3f}  '
              f'({"ESTABILIZOU" if cv_fim < cv_ini * 0.6 else "ainda oscila"})')
        print(f'  Redução de volatilidade: {(1-cv_fim/cv_ini)*100:.0f}%' if cv_ini else '')
    else:
        R_final = R[:, -100:].mean(axis=1)
        print(f'ESTABILIDADE (últimos 100 episódios, {n_seeds} seeds):')
        print(f'  Recompensa média entre seeds : {R_final.mean():>10.1f}')
        print(f'  Desvio-padrão entre seeds    : {R_final.std():>10.1f}')
        cv = R_final.std() / abs(R_final.mean()) if R_final.mean() != 0 else 0
        print(f'  Coef. de variação (CV)       : {cv:>10.3f}  '
              f'({"ESTÁVEL" if cv < 0.15 else "instável" if cv > 0.3 else "ok"})')
        print(f'  Por seed: {[f"{v:.0f}" for v in R_final]}')
    print(f'{"="*60}')
    print(f'\n✓ Gráficos salvos: {out_png}')
    return True


if __name__ == '__main__':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--iter_nome', default='final_estavel')
    args = p.parse_args()

    HERE = Path(__file__).parent
    PROJ = HERE.parent
    gerar(args.iter_nome, HERE / 'logs',
          PROJ / 'results' / 'v21' / 'curvas_treino.png')
