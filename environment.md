# Ambiente de referência

**Toda a análise** — inferência, fusão, calibração, métricas, figuras e relatório — rodou nesta
máquina, e é ela que os números do relatório descrevem.

| Item | Valor |
|---|---|
| SO | Windows 11 Home Single Language 26200 |
| GPU | NVIDIA GeForce RTX 3050 6 GB Laptop |
| Driver NVIDIA | 581.95 |
| Python | 3.11.4 |
| PyTorch | 2.9.1+cu126 (CUDA disponível) |
| Ultralytics | 8.4.72 |
| OpenCV | 4.10.0 |
| NumPy | 2.3.5 |
| Pillow | 12.2.0 |

**Dois treinos não couberam aqui.** O `yolo26n_imgsz1280_300ep` e o `yolo26m_imgsz1280_300ep_aug` da §6 foram treinados
numa **Tesla T4 de 16 GB (Google Colab)**, porque a 1280 px o YOLO26m estoura os 6 GB desta
placa. Dá para conferir sem confiar na minha palavra: `train_args["data"]` dentro dos dois
checkpoints aponta para `/content/data/...`, enquanto os demais apontam para um caminho local
do Windows. (O dataset mudou de lugar depois; o caminho gravado no `.pt` é o de quando treinou.)

Isso **não** contamina a comparação da §6, e vale explicar por quê: a T4 treinou, mas quem
mediu foi esta máquina. Os oito checkpoints passaram pelo mesmo `scripts/12_compare_models.py`,
no mesmo teste, com o mesmo pipeline de fusão — a GPU de treino não entra em nenhuma métrica
reportada. O que ela afeta é o `batch` viável — 10 no `yolo26n_imgsz1280_300ep`, contra os 4 que `11_experiments.py`
declara para caber nesta placa. Quem rodar a receita local reproduz o **experimento**, não o
arquivo binário; a ressalva está registrada no próprio spec dele.

## Determinismo

`seed: 1337` em `configs/experiment.yaml`, propagada para `random`, `numpy` e `torch`
(`torch.use_deterministic_algorithms` quando o kernel permite) por `src/utils/seed.py`.
O treino do Ultralytics roda com `deterministic: true`.

Reprodutibilidade bit-a-bit entre GPUs diferentes não é garantida (kernels cuDNN divergem);
as métricas agregadas devem reproduzir dentro do desvio entre folds reportado no relatório.

**Ressalva declarada:** o treino usa `cache: "ram"`, e o próprio Ultralytics avisa que isso
pode produzir resultados não determinísticos. A escolha foi deliberada — com `workers: 0`
(obrigatório neste Windows, ver abaixo) o cache em RAM é o que torna o treino viável em tempo,
decodificando cada PNG 720×1280 uma única vez.

**Diferença de protocolo, também declarada:** os três folds agrupados foram treinados com
`cache: "ram"`; o baseline de split aleatório foi treinado com `cache: "disk"`, via
`--cache disk`. O motivo é operacional: quando o run dele começou, a RAM livre da máquina havia
caído para 0,7 GiB e o treino passou a paginar, indo de ~20 s para 82 s por época. O cache
afeta **apenas como as imagens chegam ao dataloader** — não a arquitetura, as épocas, a
augmentação, a taxa de aprendizado nem a semente. A comparação de vazamento continua válida em
tudo que a define, mas a diferença fica registrada porque um leitor tem o direito de saber que
os dois runs não foram bit-a-bit idênticos em configuração.

## Ordem de import: torch antes de pandas

`import pandas` seguido de `import torch` falha **de forma determinística** nesta máquina, com
`OSError: [WinError 1114]` na inicialização de `c10.dll`. Medido: numpy, cv2, sklearn, scipy,
PIL e matplotlib não disparam; só o pandas. `KMP_DUPLICATE_LIB_OK=TRUE`, o paliativo mais
citado, **não resolve**.

Por isso todo script que usa GPU importa `src.utils.torch_first` como primeiro import
não-stdlib. Se um erro `WinError 1114` aparecer, a causa é quase certamente um import de
pandas que passou na frente.

## Orçamento de memória

15,7 GiB de RAM, dos quais ~10 livres em condições normais. O treino com `cache: "ram"` segura
cerca de 1,2 GiB do dataset decodificado, e o pico de VRAM chega a 5,2 dos 6 GiB.

**Não rode a geração de figuras nem a análise durante um treino.** A combinação esgotou a
memória e derrubou um treino com `numpy._core._exceptions._ArrayMemoryError` ao tentar alocar
1,17 MiB — o erro aparece minúsculo justamente porque a máquina já estava no limite.
`scripts/02_train_cv.py --resume` existe para que uma interrupção assim custe minutos e não a
corrida inteira.

**O número a vigiar é o *commit* livre, não a RAM livre.** Um segundo treino caiu com
`cv::OutOfMemoryError` do OpenCV ao alocar **1.228.800 bytes** — exatamente um buffer de
640×640×3. A RAM livre marcava 0,7 GiB, valor que já tinha oscilado a noite toda sem incidente;
o que estava de fato esgotado era o **commit** (1,5 GiB). É o commit que decide se um `malloc`
falha, e ele despencou porque outro processo — um navegador com 2,6 GiB — disputava a máquina.
Nas execuções seguintes, com 5 a 15 GiB de commit livre, nenhuma queda.

**`scripts/11_experiments.py` usa `cache: "disk"`, não `"ram"`.** Com o dataset de recortes
(10× mais imagens que o original) e a 1280 px, o cache em RAM não cabe. A troca custa I/O só na
primeira época — 252 s contra os 53 s das seguintes, que é a construção do cache — e depois o
tempo por época volta ao normal. Pico de VRAM a 1280 px: **5,6 dos 6 GiB**, com `batch: 4`.

## Onde ficam os dados

Dataset e saídas de treino ficam em **`ml/`**, dentro da pasta do projeto: são **7,5 GB** em
17 mil arquivos (o MinneApple, o cache `.npy` que o Ultralytics gera no treino, e os `runs/` do
Ultralytics). Ficam ali para tudo viver num lugar só, e **`/ml/` está no `.gitignore`** — nada
disso entra no repositório, que tem 101 arquivos.

Os caminhos estão em `configs/paths.yaml` e são **relativos à raiz do repositório**, então
funcionam de qualquer diretório de trabalho. É o único arquivo a ajustar noutra máquina: ou
aponte `dataset_root` para onde você extraiu o dataset, ou recrie `ml/minneapple` seguindo
`scripts/00_download.md`.
