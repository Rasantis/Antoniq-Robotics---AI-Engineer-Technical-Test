# Checkpoints

Os oito conjuntos de pesos da §6 do relatório, versionados aqui para que a tabela de
comparação possa ser reproduzida — e uma inferência rodada — sem baixar os 1,9 GB do dataset
e treinar antes. São 126 MB; é a única exceção à regra de não versionar binário.

Todos treinados no **fold 0** (364 imagens de treino, 81 de validação), semente 1337. A contagem
de parâmetros abaixo foi **medida** nos próprios arquivos, não copiada da tabela do Ultralytics.

Dois deles — `yolo26n_imgsz1280_300ep` e `yolo26m_imgsz1280_300ep_aug` — foram treinados numa **Tesla T4 de 16 GB**,
porque a 1280 px não cabem nos 6 GB da placa local (ver `environment.md`). Por isso
`results/experiments.json`, que é o registro da fila **local**, tem cinco entradas e não oito.
Quem mediu os oito foi esta máquina, pelo mesmo `scripts/12_compare_models.py`.

**Como ler o nome.** `<arquitetura>_imgsz<entrada>_<orçamento>ep[_aug]` — é a receita que
reproduz o treino. O número de épocas do nome é o **orçamento pedido**; a coluna *rodou* abaixo
é onde a `patience` de fato parou. Em seis dos oito o early stopping disparou antes, e é por isso
que as duas colunas existem: o nome diz o que rodar, a tabela diz o que aconteceu.

| arquivo | arquitetura | params | entrada | orçamento | rodou | augmentação | MAE no teste |
|---|---|---|---|---|---|---|---|
| **`yolo26s_imgsz1280_200ep_aug.pt`** | YOLO26s | 9,9 M | 1280 | 200 | 56 | anti-deriva | **12,6** |
| `yolo26s_imgsz1280_200ep.pt` | YOLO26s | 9,9 M | 1280 | 200 | 63 | padrão | 15,1 |
| `yolo26m_imgsz1280_300ep_aug.pt` | YOLO26m | 21,8 M | 1280 | 300 | 92 | anti-deriva | 19,3 |
| `yolo26n_imgsz1280_300ep.pt` | YOLO26n | 2,5 M | 1280 | 300 | 124 | padrão | 19,1 † |
| `yolo26n_imgsz1280_90ep.pt` | YOLO26n | 2,5 M | 1280 | 90 | 90 | padrão | 24,7 |
| `yolo26n_tiles320_15ep.pt` | YOLO26n | 2,5 M | 640 ‡ | 15 | 15 | padrão | 26,6 |
| `yolo11s_imgsz640_120ep.pt` | YOLO11s | 9,4 M | 640 | 120 | 62 | padrão | 33,0 |
| `yolo26n_imgsz640_120ep.pt` | YOLO26n | 2,5 M | 640 | 120 | 80 | padrão | 33,2 |

† No braço eleito pela validação (`A_full640`) a MAE é 31,1; os 19,1 são do `B_full1280`.
A §6 discute essa divergência: 0,38 maçã de diferença na validação custou doze pontos de MAE.

‡ Treinado sobre recortes de **320 px** alimentados à rede em 640 — por isso o nome traz
`tiles320` no lugar de `imgsz`: a entrada da rede não é o que distingue este treino.

**A augmentação "anti-deriva"** é `hsv_v=0.75, hsv_s=0.90, scale=0.35, close_mosaic=30`, e o
motivo é um número medido sobre as 28.182 instâncias: o **V médio dentro das caixas anotadas**
é **98,2** nas seis sessões de treino do fold 0 (de 54 a 129 por sessão) e **164,2 / 166,3** nas
duas sessões de 19/09 — que são justamente as que o modelo mais erra. Como o `hsv_v` multiplica
V por um ganho em `[1−g, 1+g]`, o default `g=0,4` leva a média de treino no máximo a
98,2 × 1,40 = **137**: não alcança 164. Com `g=0,75` vai a **172**, e alcança.

Não confunda com o brilho **por imagem** de `results/images.csv`, que vai de 100 a 143 por
sessão: aquele inclui céu e folhagem, este é só o pixel da fruta.

## Uso

```bash
# inferência: usa yolo26s_imgsz1280_200ep_aug.pt automaticamente se não houver pesos treinados localmente
python scripts/run_inference.py --image caminho/da/imagem.png

# um checkpoint específico
python scripts/run_inference.py --image img.png --weights models/yolo26m_imgsz1280_300ep_aug.pt

# a tabela comparativa, a partir destes arquivos
python scripts/12_compare_models.py
```

As configurações de treino de cada um estão em `scripts/11_experiments.py`.
