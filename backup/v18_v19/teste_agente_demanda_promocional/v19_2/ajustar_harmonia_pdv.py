"""Ajusta a matriz harmonia da calibração para refletir a SEMÂNTICA DE COMBO
NO BALCÃO DE POSTO DE GASOLINA.

Insight do Vinicius (15/05/2026): a matriz herdou números da Instacart (cesta
de supermercado) e dizia gelo+destilados=2.20. Mas saco de gelo 5kg é compra
de OCASIÃO (festa em casa); garrafa de whisky no posto é compra INDIVIDUAL.
Cliente não leva 5kg de gelo + 1 garrafa de whisky no balcão.

REGRA: gelo só faz sentido com cerveja (ambos vão para o churrasco/festa).
       gelo + qualquer outra coisa = harmonia neutra ou baixa.

Este script roda 1× e modifica calibracao_v2.json da V19.2 em loco.
"""
import json
from pathlib import Path


def ajustar(path_calibracao: str):
    with open(path_calibracao, encoding='utf-8') as f:
        cal = json.load(f)

    cats = [c['categoria'] for c in cal['categorias']]
    idx = {c: i for i, c in enumerate(cats)}
    harm = cal['harmonia_combo']

    # Mudanças cirúrgicas (sempre simétricas: matriz[a][b] = matriz[b][a])
    ajustes = [
        # PAR             NOVO VALOR  COMENTÁRIO
        (('gelo', 'destilados'),    0.80, 'Saco 5kg não casa com garrafa individual no balcão'),
        (('gelo', 'vinho'),          0.80, 'Idem destilados'),
        (('gelo', 'refrigerante'),   0.95, 'Raro alguém levar 5kg gelo + refri'),
        (('gelo', 'suco'),           0.90, 'Não faz sentido balcão'),
        (('gelo', 'isotonico'),      0.90, 'Não faz sentido balcão'),
        (('gelo', 'sorvete'),        0.90, 'Cliente leva sorvete ou gelo, não os dois'),
        (('gelo', 'energetico'),     0.90, 'Não casa'),
        # gelo + cerveja = MANTÉM 2.40 (churrasco clássico)
        # gelo + snack = MANTÉM 1.40 (cliente pode levar gelo + amendoim pra churrasco)
        # Outros pares óbvios que não casam em PDV:
        (('cafe', 'cerveja'),        0.70, 'Combo absurdo (já era penalizado no agente)'),
        (('cafe', 'destilados'),     0.70, 'Idem'),
    ]

    print('=' * 80)
    print('Ajustes em matriz_harmonia (calibracao_v2.json V19.2)')
    print('=' * 80)
    for (a, b), novo, comentario in ajustes:
        if a not in idx or b not in idx:
            print(f'  ! {a} ou {b} não está no catálogo, ignorando')
            continue
        ia, ib = idx[a], idx[b]
        antigo = harm[ia][ib]
        harm[ia][ib] = novo
        harm[ib][ia] = novo  # simétrica
        print(f'  {a:<14s} + {b:<14s}  {antigo:>5.2f} → {novo:>4.2f}   ({comentario})')

    # Salva
    with open(path_calibracao, 'w', encoding='utf-8') as f:
        json.dump(cal, f, ensure_ascii=False, indent=2)
    print()
    print(f'✓ Matriz salva em {path_calibracao}')


if __name__ == '__main__':
    import io, sys
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    here = Path(__file__).parent
    ajustar(str(here / 'calibracao_v2.json'))
