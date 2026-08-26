"""Etapa 8: monta o relatório em HTML pronto para impressão em PDF.

O entregável pedido é um PDF de 4 a 6 páginas. Este script converte `report/relatorio.md` num
HTML com CSS de impressão em A4 — margens, quebras de página controladas, tabelas que não se
partem no meio — e imprime a contagem estimada de páginas para que o texto possa ser ajustado
ao limite antes de gerar o PDF.

A conversão para PDF em si é feita pelo navegador (Ctrl+P → Salvar como PDF). A alternativa
seria WeasyPrint, que está instalado nesta máquina mas sem as bibliotecas nativas de que
depende no Windows; e depender de uma cadeia de ferramentas frágil num entregável que precisa
ser reproduzível pelo avaliador é pior do que uma ação manual de um clique.

Uso:
    python scripts/08_report.py
    python scripts/08_report.py --open      # abre no navegador ao terminar
"""
from __future__ import annotations

import argparse
import base64
import re
import sys
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import markdown  # noqa: E402

# ~450 palavras por página A4 com esta tipografia; tabelas e figuras entram como área.
WORDS_PER_PAGE = 450
FIGURE_PAGE_FRACTION = 0.32  # fração de página que uma figura ocupa, em média

CSS = """
@page { size: A4; margin: 10mm 13mm; }
:root {
  --ink: #14140f; --ink-soft: #4a4a44; --rule: #d8d7d0;
  --accent: #2a78d6; --surface-alt: #f7f7f4;
}
* { box-sizing: border-box; }
body {
  font-family: "Charter", "Georgia", "Source Serif Pro", serif;
  font-size: 9.2pt; line-height: 1.21; color: var(--ink);
  max-width: 178mm; margin: 0 auto; padding: 0; hyphens: auto;
}
h1 { font-size: 19pt; line-height: 1.2; margin: 0 0 2mm; letter-spacing: -0.01em; }
h1 + p { color: var(--ink-soft); font-size: 10.5pt; margin: 0 0 6mm; }
h2 {
  font-size: 12pt; margin: 3.8mm 0 1.7mm; padding-bottom: 1.2mm;
  border-bottom: 1.6px solid var(--accent); break-after: avoid;
}
h3 { font-size: 10.4pt; margin: 3mm 0 1.2mm; color: var(--ink); break-after: avoid; }
p { margin: 0 0 1.7mm; text-align: justify; }
ul, ol { margin: 0 0 2.6mm; padding-left: 5mm; }
li { margin-bottom: 0.5mm; }
strong { font-weight: 700; }
code {
  font-family: "Consolas", "DejaVu Sans Mono", monospace; font-size: 8.6pt;
  background: var(--surface-alt); padding: 0.3mm 1mm; border-radius: 2px;
}
pre {
  background: var(--surface-alt); border-left: 2.5px solid var(--accent);
  padding: 2.2mm 3mm; font-size: 8.4pt; line-height: 1.34; overflow-x: auto;
  break-inside: avoid; margin: 0 0 3mm;
}
pre code { background: none; padding: 0; }
table {
  border-collapse: collapse; width: 100%; margin: 0 0 2.4mm;
  font-size: 8.2pt;
}
thead { display: table-header-group; }   /* cabecalho repete se a tabela quebrar */
tr { break-inside: avoid; }              /* uma LINHA nunca parte ao meio */
th {
  text-align: left; font-weight: 700; border-bottom: 1.4px solid var(--ink);
  padding: 0.8mm 1.6mm; background: var(--surface-alt);
}
td { padding: 0.7mm 1.6mm; border-bottom: 0.6px solid var(--rule); }
tr:last-child td { border-bottom: 1.2px solid var(--ink); }
td:not(:first-child), th:not(:first-child) { text-align: right; }
blockquote {
  margin: 0 0 2.8mm; padding: 2mm 3.2mm; background: var(--surface-alt);
  border-left: 2.5px solid var(--accent); break-inside: avoid; font-size: 9.6pt;
}
blockquote p:last-child { margin-bottom: 0; }
figure { margin: 2mm 0 2.5mm; break-inside: avoid; text-align: center; }
/* Teto de altura: uma figura nunca deve ocupar mais que ~40% de uma pagina util (263 mm).
   Sem isso, um painel retrato empurra o relatorio para fora do limite de paginas sozinho. */
figure img, p > img {
  /* O markdown emite <p><img>, nao <figure><img> — o seletor precisa cobrir os dois,
     senao o teto de altura nao e aplicado e a imagem renderiza na altura natural. */
  display: block; margin: 0 auto; max-width: 100%; height: auto; max-height: 48mm;
}
figcaption {
  font-size: 8.4pt; color: var(--ink-soft); margin-top: 1.2mm;
  text-align: left; font-style: italic;
}
hr { display: none; }   /* os <h2> ja separam as secoes; a regra so gastava espaco */
em { color: var(--ink-soft); }
"""


def embed_figures(html: str, base: Path) -> tuple[str, int]:
    """Troca <img src="..."> por data URI, para o HTML ser um arquivo único.

    Um relatório que só renderiza se a pasta de figuras estiver do lado é um relatório que
    chega quebrado por e-mail.
    """
    count = 0

    def replace(match: re.Match) -> str:
        nonlocal count
        src = match.group(1)
        # Caminhos relativos sao resolvidos contra a pasta do MARKDOWN, nao a do repo:
        # e assim que o link se comporta quando alguem abre o .md direto.
        path = (base / src).resolve() if not Path(src).is_absolute() else Path(src)
        if not path.exists():
            print(f"  AVISO: figura ausente, mantida como referência: {src}")
            return match.group(0)
        data = base64.b64encode(path.read_bytes()).decode()
        count += 1
        return f'src="data:image/png;base64,{data}"'

    return re.sub(r'src="([^"]+)"', replace, html), count


def estimate_pages(md_text: str, n_figures: int) -> float:
    """Estimativa grosseira, só para saber se o texto cabe no limite de 4 a 6 páginas."""
    prose = re.sub(r"```.*?```", "", md_text, flags=re.S)
    prose = re.sub(r"^\|.*$", "", prose, flags=re.M)   # tabelas contam à parte
    words = len(prose.split())
    table_rows = len(re.findall(r"^\|", md_text, flags=re.M))
    return words / WORDS_PER_PAGE + table_rows / 46 + n_figures * FIGURE_PAGE_FRACTION


BROWSERS = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]


def to_pdf(html: Path) -> Path | None:
    """Imprime o HTML em PDF com um navegador headless.

    Chrome e Edge têm um motor de impressão que respeita `@page` e `break-inside`, que é o que
    o layout precisa. WeasyPrint seria a opção pura-Python, mas nesta máquina ele está
    instalado sem as bibliotecas nativas de que depende no Windows.
    """
    import subprocess

    browser = next((b for b in BROWSERS if Path(b).exists()), None)
    if browser is None:
        return None

    pdf = html.with_suffix(".pdf")
    pdf.unlink(missing_ok=True)
    subprocess.run(
        [browser, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
         "--virtual-time-budget=20000", f"--print-to-pdf={pdf}", html.as_uri()],
        capture_output=True, timeout=180,
    )
    return pdf if pdf.exists() else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--open", action="store_true", help="abre no navegador ao terminar")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    source = repo / "report" / "relatorio.md"
    if not source.exists():
        raise SystemExit(f"{source} nao existe.")

    md_text = source.read_text(encoding="utf-8")
    if "PENDENTE" in md_text:
        pending = md_text.count("PENDENTE")
        print(f"  AVISO: {pending} marcador(es) PENDENTE ainda no relatório")

    body = markdown.markdown(
        md_text, extensions=["tables", "fenced_code", "attr_list", "sane_lists"]
    )
    body = re.sub(
        r'<p>(<img[^>]*alt="([^"]*)"[^>]*/?>)</p>',
        lambda m: f'<figure>{m.group(1)}'
                  + (f'<figcaption>{m.group(2)}</figcaption>' if m.group(2) else '')
                  + '</figure>',
        body,
    )
    body, embedded = embed_figures(body, source.parent)

    title = re.search(r"^#\s+(.+)$", md_text, flags=re.M)
    html = (
        "<!doctype html>\n<html lang='pt-BR'>\n<head>\n<meta charset='utf-8'>\n"
        f"<title>{title.group(1) if title else 'Relatório'}</title>\n"
        f"<style>{CSS}</style>\n</head>\n<body>\n{body}\n</body>\n</html>\n"
    )

    out = repo / "report" / "relatorio.html"
    out.write_text(html, encoding="utf-8")

    pages = estimate_pages(md_text, embedded)
    print(f"  {out}")
    print(f"  {embedded} figuras embutidas | {len(md_text.split()):,} palavras")
    print(f"  estimativa: {pages:.1f} páginas", end="")
    print("  (dentro do limite de 4 a 6)" if 4 <= pages <= 6
          else f"  <- FORA do limite de 4 a 6 páginas")
    print("\n  Para gerar o PDF: abra o HTML no navegador e use Ctrl+P -> Salvar como PDF")
    print("  (margens padrão, A4, sem cabeçalho/rodapé)")

    pdf = to_pdf(out)
    if pdf:
        import pypdf

        n = len(pypdf.PdfReader(str(pdf)).pages)
        verdict = "dentro do limite" if 4 <= n <= 6 else "FORA do limite de 4 a 6"
        print(f"\n  {pdf}")
        print(f"  PAGINAS REAIS: {n}  ({verdict})")
    else:
        print("\n  Nenhum navegador encontrado para gerar o PDF automaticamente.")
        print("  Abra o HTML e use Ctrl+P -> Salvar como PDF.")

    if args.open:
        webbrowser.open(out.as_uri())


if __name__ == "__main__":
    main()
