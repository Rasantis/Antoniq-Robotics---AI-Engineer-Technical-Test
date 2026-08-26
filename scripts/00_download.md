# Obtenção do dataset

## Fonte oficial (a única usada neste trabalho)

MinneApple — Häni, Roy & Isler (2019), University of Minnesota Digital Conservancy.

- Página: <https://conservancy.umn.edu/handle/11299/206575>
- Artigo: <https://arxiv.org/abs/1909.06441>
- Licença: **CC BY-NC-SA 3.0 US** (não-comercial — ver § Limitações no relatório)

Baixe `detection.tar.gz` e extraia de modo a obter:

```
D:\ml\minneapple\detection\
├── train\
│   ├── images\      670 PNG 720x1280   (rotuladas, 2015, 10 sessões)
│   └── masks\       670 PNG            (máscaras de instância, 1 ID por maçã)
└── test\
    └── images\      331 PNG            (2016 — SEM rótulos, retidos no CodaLab)
```

> **Orientação das imagens.** Os arquivos são **retrato: 720 de largura por 1280 de altura**,
> verificado nos 1001 PNG. O artigo escreve "1280 x 720", que é altura x largura — uma
> inversão fácil de propagar para dentro de todo transform de coordenada. `01_prepare_data.py`
> falha alto se a resolução não for a esperada.

O caminho é configurável em `configs/paths.yaml` (`dataset_root`).

> O download automatizado não funciona: o servidor da UMN está atrás de um WAF da Azure que
> devolve 403 para qualquer cliente que não seja um navegador. O download é manual, por isso
> `scripts/01_prepare_data.py` valida a integridade do que foi extraído (contagem de arquivos,
> resolução, pareamento imagem↔máscara) antes de qualquer outra etapa.

## Mirrors de terceiros — deliberadamente NÃO usados

Existem cópias no HuggingFace e no Kaggle. A mais completa
(`lauesa1/minne-apple-segmentation`) foi inspecionada e **rejeitada**: o `test.json` declara
`"contributor": "Script by Gemini"` e traz anotações para o split de **teste**, cujos rótulos
oficiais não são públicos. São anotações sintéticas apresentadas como ground truth — usá-las
produziria métricas sem sentido para o conjunto que mais importa.

(A resolução `720x1280` que esse mirror registra está **correta**; a inversão está no texto do
artigo, não no mirror. A rejeição é pela procedência das anotações, não pelos metadados.)

Este trabalho usa **apenas as 670 imagens rotuladas de treino**, re-splitadas por sessão de captura
(ver `src/data/splits.py`).
