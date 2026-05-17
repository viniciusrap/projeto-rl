# Como funciona o sistema de promoções — explicação simples

> Documento para quem não é técnico. Explica, em português direto, como o
> computador decide quais promoções o posto deve fazer.

---

## 1. O que esse sistema faz, em uma frase

Ele olha o catálogo do posto, o calendário do ano (feriados, Copa, Dia das
Mães...), o clima e o histórico de vendas, e **monta um calendário de
promoções** dizendo: *"no dia X, faça promoção do produto Y com o produto Z"*.

---

## 2. A ideia central: o computador "aprende" como um funcionário novo

Imagine um funcionário novo que **não sabe nada** sobre o posto. No primeiro
dia ele chuta promoções aleatórias: "vamos dar desconto em whisky numa terça
de manhã". No fim do dia, o gerente diz: *"deu prejuízo, foi ruim"*.

No dia seguinte ele tenta de novo. Aos poucos ele percebe padrões:
- "Cerveja + amendoim na sexta à noite **vende muito** → o gerente elogia"
- "Desconto em água numa segunda de manhã **não muda nada** → o gerente reclama"
- "Promoção de gelo + whisky **ninguém leva junto** → prejuízo"

Depois de **milhares de dias simulados**, o funcionário vira um especialista:
ele já sabe, sem pensar, qual promoção fazer em cada situação. Esse
"funcionário" é o nosso **agente de Reinforcement Learning (RL)**.

**Reinforcement Learning = aprender por tentativa, erro e recompensa.**
Igual treinar um cachorro: acertou, ganha petisco; errou, não ganha. Com o
tempo ele só faz o que dá petisco.

---

## 3. As 3 decisões que o agente toma (os "3 mini-agentes")

Toda vez que aparece uma oportunidade de promoção, o agente decide **3
coisas ao mesmo tempo**. É como ter 3 especialistas trabalhando juntos:

| Mini-agente | O que decide | Exemplo |
|---|---|---|
| **Agente de Desconto** | Faz promoção? Quanto? | "Combo" ou "−10%" ou "não fazer nada" |
| **Agente de Combo** | Com qual produto juntar? | "Cerveja **com salgadinho**" |
| **Agente de Margem** | O desconto vai em qual produto? | "Desconto no salgadinho, cerveja preço cheio" |

Eles compartilham a mesma "leitura da situação" e decidem em conjunto.

---

## 4. O que o agente "enxerga" antes de decidir (o estado)

Antes de cada decisão, o agente recebe um "raio-x" da situação com ~90
informações. Em linguagem simples, ele sabe:

- **Que dia é**: dia da semana, mês, se é véspera de feriado
- **Que evento está chegando**: Dia das Mães em 5 dias? Copa amanhã?
- **Qual produto está na mão**: cerveja? café? chocolate?
- **Como esse produto vende normalmente**: muito na sexta, pouco na segunda
- **Situação do estoque**: tem muito parado? está perto de vencer?
- **O clima**: calor (gelo/sorvete vendem mais) ou frio
- **O que já foi promovido essa semana**: pra não repetir sempre a mesma coisa

---

## 5. Como ele sabe se acertou (a recompensa)

Depois de cada decisão, o sistema calcula uma **nota**. A nota sobe quando a
promoção:

- ✅ Dá **lucro de verdade** (vende mais sem destruir a margem)
- ✅ Junta produtos que **combinam** (cerveja+amendoim, café+pão)
- ✅ Acerta o **evento certo** (chocolate no Dia das Mães)
- ✅ É a **Esquenta de Sexta** (cerveja+salgadinho sexta/sábado)
- ✅ **Gira produto parado** ou perto de vencer

E a nota **despenca** quando:

- ❌ Junta produtos que **não casam** (gelo + whisky — ninguém leva no posto)
- ❌ Dá **desconto em produto que já vende sozinho** (perde margem à toa)
- ❌ Promove algo de **giro quase zero** (whisky, vinho)
- ❌ **Repete a mesma promoção** muitas vezes (cliente enjoa)
- ❌ Dá **prejuízo**

O agente passa o treino inteiro tentando tirar a maior nota possível. No fim,
a estratégia que tira nota alta é exatamente a que faz **bom negócio para o
posto**.

---

## 6. Por que a gente NÃO usou dados de supermercado

No começo o sistema usava dados de um supermercado gigante (Instacart). O
problema: no supermercado a pessoa enche o carrinho pra uma festa em casa —
**gelo + whisky + refrigerante** faz sentido lá. Mas **num posto não**: quem
para pra abastecer e leva uma cerveja gelada não vai sair com um saco de
gelo de 5kg.

Por isso recriamos a "tabela de combinações" do zero, pensando **só no balcão
de um posto de gasolina**:

| Combina no posto | Não combina no posto |
|---|---|
| Café + pão de queijo (manhã) | Gelo + whisky (isso é mercado) |
| Cerveja + amendoim (noite/sexta) | Café + cerveja (manhã vs noite) |
| Refrigerante + salgadinho (tarde) | Padaria + cerveja |
| Chocolate caixa + balas (presente) | Vinho + qualquer coisa (não vende em posto) |

---

## 7. O que o agente aprendeu sozinho (resultados)

Depois do treino, olhando o calendário de 1 ano que ele montou:

- **Esquenta de Sexta**: toda sexta/sábado ele recomenda **Cerveja +
  Salgadinho**. Foi o combo nº 1 do ano (apareceu ~45 vezes). Ele descobriu
  isso sozinho — ninguém programou "faça isso toda sexta".
- **Copa do Mundo**: nos dias de jogo do Brasil, **Cerveja + Salgadinho**
  pra galera assistir.
- **Datas comerciais**: Dia das Mães/Namorados/Pais/Crianças → combo de
  **Chocolate (caixa)** como presente.
- **Manhã**: **Café + Biscoito/Pão** pro trabalhador que para no caminho.
- **Misturou combos e descontos diretos**: ~60% combos, ~40% descontos
  individuais (em produto parado ou de bastante volume).
- **Não recomenda nada absurdo**: zero combos tipo gelo+whisky.

---

## 8. Como usar na prática (passo a passo)

```
1. Atualizar os dados do posto (inventário, descarte)
       ↓
2. Rodar:  construir_calibracao_pdv.py   (monta a "ficha" de cada produto)
       ↓
3. Rodar:  iterar_v21.py                 (treina o agente — ~15 min)
       ↓
4. Rodar:  gerar_calendario_v21.py       (o agente monta o calendário do ano)
       ↓
5. Rodar:  gerar_calendario_visual.py    (gera o calendário bonito em HTML)
       ↓
6. Abrir:  results/v21/calendario_visual.html
       → o dono vê o ano todo, mês a mês, e decide o que aplicar
```

O dono **não precisa seguir 100%** do calendário. Ele é uma **recomendação**:
o dono olha, vê o que faz sentido pro movimento dele, e aplica.

---

## 9. Limites honestos (o que o sistema NÃO garante)

- Os valores de lucro (R$ X) são **estimativas do modelo**, não medições
  reais. Para confirmar, seria preciso testar no posto de verdade
  (fazer a promoção uma semana sim, uma não, e comparar).
- O sistema assume que o desconto aumenta a venda numa proporção tirada da
  literatura de varejo. Cada posto é diferente — os números reais só
  aparecem testando no balcão.
- O valor real desse trabalho é o **plano de promoções coerente** e o
  **método** (que pode ser recalibrado quando chegarem dados reais), não o
  número exato de reais.

---

## 10. Resumo de uma linha

> Um "funcionário virtual" treinado em milhares de dias simulados aprendeu,
> por tentativa e erro, qual promoção dá mais lucro em cada dia do ano para
> a conveniência do posto — e entrega isso num calendário pronto para o dono
> usar.

---

*Auto Posto Parque Viana · Barueri/SP · Projeto de Reinforcement Learning
(Insper) — Vinicius Rocha e Luigi Zema*
