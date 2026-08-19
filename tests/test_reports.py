import pandas as pd

from econometria.reports import ReportBuilder


def test_report_builder_render_html_contem_conteudo():
    builder = ReportBuilder("Relatorio de Teste", "Subtitulo de teste")
    builder.add_card("Indicador", "42")
    builder.add_section(
        "Secao 1",
        "**Negrito** e texto normal.\n\n- item um\n- item dois",
        tabelas=[("Tabela X", pd.DataFrame({"a": [1, 2], "b": [3, 4]}))],
    )
    html = builder.render_html()
    assert "Relatorio de Teste" in html
    assert "Indicador" in html
    assert "<strong>Negrito</strong>" in html
    assert "<li>item um</li>" in html
    assert "Tabela X" in html


def test_report_builder_to_docx_bytes_gera_arquivo_valido():
    builder = ReportBuilder("Relatorio DOCX", "")
    builder.add_section("Secao", "Texto simples.", tabelas=[("Tab", pd.DataFrame({"x": [1, 2]}))])
    conteudo = builder.to_docx_bytes()
    assert conteudo[:2] == b"PK"  # docx e um zip
    assert len(conteudo) > 100


def test_report_builder_to_excel_bytes_gera_arquivo_valido():
    builder = ReportBuilder("Relatorio XLSX", "")
    builder.add_section("Secao", "", tabelas=[("Tab", pd.DataFrame({"x": [1, 2], "y": [3, 4]}))])
    conteudo = builder.to_excel_bytes()
    assert conteudo[:2] == b"PK"  # xlsx tambem e um zip
    assert len(conteudo) > 100
