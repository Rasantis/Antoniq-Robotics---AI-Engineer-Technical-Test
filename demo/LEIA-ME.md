# Console de defesa

Ferramenta de apresentação — o enunciado não pede isto. Ela importa o pipeline de `src/`, então
o que a tela mostra é o mesmo código que gerou o relatório.

## Antes de entrar na sala

```bash
pip install -r demo/requirements.txt     # só o Flask; uma vez só
python demo/app.py
```

Abre em <http://127.0.0.1:5000>. Duas coisas para fazer **antes** de começar:

1. Aperte `D` e clique em **Aquecer o modelo**. A primeira inferência carrega os pesos e leva
   ~25 s (medido); depois de aquecido são ~2,5 s para os quatro braços. Não faça isso na
   frente deles. Se trocar de pesos no seletor, aqueça o outro também — cada um carrega uma vez.
2. Aperte `F11` no navegador para tela cheia.

## Navegar sob pergunta

O trilho de cima é o índice. Cada estação tem uma tecla — se perguntarem sobre duplicata, você
aperta `5` e já está na ablação de fusão.

| tecla | estação | responde a |
|---|---|---|
| `1` | Tese | "me conta o que você fez" |
| `2` | Dataset | "por que MinneApple?" · "quantas imagens?" |
| `3` | Split | "como você garantiu que não vazou?" |
| `4` | Braços | "o tiling ajudou?" · "e o custo?" |
| `5` | Fusão | "como você tratou duplicata?" · "por que não IoU?" |
| `6` | Erros | "onde o modelo erra?" |
| `7` | Calibração | "como escolheu o limiar?" |
| `8` | Correção | "o que você fez com a análise de erro?" |
| `9` | Go/no-go | "isso está pronto para produção?" |
| `0` | Produção | "o que faria com mais tempo?" · "o que falta para ir a produção?" |
| `D` | **Demo ao vivo** | "mostra funcionando" |
| `S` | Siglas | consulta rápida, se travar num termo |

Setas navegam. A linha de baixo mostra **de qual arquivo** sai o número da tela — se perguntarem
"de onde vem esse valor?", a resposta já está projetada.

## O demo

Solte uma imagem, cole com `Ctrl+V`, ou clique nos botões de exemplo. Roda os **quatro braços**
na mesma imagem.

**Braço não é modelo.** Os quatro cartões rodam **os mesmos pesos** — o nome deles aparece
numa linha acima da grade. O que muda entre os cartões é a *estratégia de inferência*, e o
código no canto (`A_full640`, `B_full1280`, `C_tile640`, `D_tile320`) é o rótulo da §4 do
relatório, para você ligar a tela à tabela. Cada cartão diz em português o que faz e com
quantos pixels a maçã mediana chega à rede — e é aí que se vê o argumento da §4: o `C` entrega
os **mesmos 27,3 px** do `B` e custa **6 passes em vez de 1**. Recortar em 640 e alimentar a
rede em 640 não magnifica nada.

**A verdade anotada aparece na tela.** Se a imagem for uma das 670 do MinneApple, o console a
reconhece (pelo sha1 do arquivo, ou pelo nome se ela foi re-salva) e mostra a contagem anotada,
o **erro com sinal** de cada braço e a partição **TP / FP / FN**. Se for uma foto qualquer, ele
diz que não há anotação em vez de inventar uma.

As caixas saem em quatro cores, as mesmas das figuras do relatório:

| cor | significa |
|---|---|
| verde | **TP** — o modelo achou a maçã anotada |
| vermelho | **FP** — apontou onde não há anotação |
| âmbar | **FN** — a anotação que ele perdeu |
| azul, fino por dentro | a **anotação**, em todas as 75 — mostra onde a fruta realmente está |

**Clique na imagem** para abrir em resolução nativa. É necessário: no painel a foto aparece com
~140 px de largura e uma maçã de 27 px vira um ponto — a cor só comunica ampliada.

Dois roteiros que funcionam bem:

**Mostrar o mecanismo do tiling.** Use a sessão fácil. O `C_tile640` gera 148 caixas brutas e
entrega 51 — ele remove 97 duplicatas. É a §4 acontecendo na tela: o tiling não fabrica
detecção, ele fabrica duplicata, e quem resolve é a política de fusão.

**Mostrar por que MAE não é critério de aceitação.** Ainda na sessão fácil (verdade **75**), o
`D_tile320` marca **75 — erro zero**. Parece perfeito. Abra a lupa: **54 TP, 21 FP, 21 FN**. Ele
acerta porque 21 falsos positivos cancelam exatamente 21 falsos negativos. É a identidade
`predito − real = FP − FN` na tela, e é o argumento inteiro da §3 em um clique.

**Mostrar o ganho do modelo.** Use a sessão difícil (19/09, a que a §5 aponta como dominante no
erro) e alterne os pesos no seletor. O baseline acha **4** maçãs; o YOLO26s @1280 + aug acha **28**. Mesma imagem,
mesmo pipeline.

Cada painel diz de onde veio o limiar: *os próprios pesos* quando a Etapa 12 mediu aquele par
(modelo, braço), *ponto congelado* quando caiu no ponto do pipeline. Isso importa — sem essa
distinção a tela contradiria a §6.

## Se algo der errado

- **O demo não responde:** os pesos não carregaram. Veja o terminal onde o `app.py` está rodando.
- **Arrastar não funciona:** use os botões de exemplo (`demo/exemplos/`). Eles não vão no git —
  são imagens do MinneApple, que é CC BY-NC-SA — mas estão no disco desta máquina.
- **A verdade não aparece:** a imagem não é uma das 670, ou `demo/gt_index.csv` sumiu.
  Reconstrua com `python demo/indexar_gt.py` (precisa do dataset em `ml/minneapple`).
- **Porta ocupada:** `python demo/app.py --port 8080`.
- **Sem internet:** não faz diferença, nada aqui usa CDN.

## Uma nota de licença

`demo/gt_index.csv` é versionado e leva as **28.182 caixas anotadas** do MinneApple — coordenadas,
não pixels. As *imagens* ficam fora do repositório de propósito (`demo/exemplos/` está no
`.gitignore`), porque redistribuir a foto é redistribuir a obra; as coordenadas são dado derivado,
e vão junto para o console funcionar numa máquina sem o dataset.

O MinneApple é **CC BY-NC-SA 3.0 US**: atribuição (o relatório e o `README` citam a fonte), uso
não comercial (é um assessment) e *share-alike*. Se preferir não redistribuir nem as coordenadas,
acrescente `demo/gt_index.csv` ao `.gitignore` e gere localmente com `python demo/indexar_gt.py` —
o console degrada com elegância e diz que não há anotação para comparar.
