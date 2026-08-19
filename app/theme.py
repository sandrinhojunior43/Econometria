"""Identidade visual compartilhada por todas as paginas do sistema Econometria."""
from __future__ import annotations

import streamlit as st

CORES = {
    "azul_900": "#0b2545",
    "azul_700": "#13315c",
    "azul_500": "#2563eb",
    "azul_100": "#e8f0fe",
    "verde_600": "#0f9d58",
    "vermelho_600": "#d93025",
    "ambar_600": "#b7791f",
    "cinza_50": "#f8f9fb",
    "cinza_100": "#eef1f6",
    "cinza_300": "#d4dae3",
    "cinza_600": "#5b6677",
    "cinza_900": "#1a2233",
}

_CSS = f"""
<style>
:root {{
    --azul-900: {CORES['azul_900']};
    --azul-700: {CORES['azul_700']};
    --azul-500: {CORES['azul_500']};
    --azul-100: {CORES['azul_100']};
    --verde-600: {CORES['verde_600']};
    --vermelho-600: {CORES['vermelho_600']};
    --ambar-600: {CORES['ambar_600']};
    --cinza-50: {CORES['cinza_50']};
    --cinza-100: {CORES['cinza_100']};
    --cinza-300: {CORES['cinza_300']};
    --cinza-600: {CORES['cinza_600']};
    --cinza-900: {CORES['cinza_900']};
}}

html, body, [class*="css"] {{
    font-family: "Segoe UI", -apple-system, BlinkMacSystemFont, Roboto, Arial, sans-serif;
}}

.bloco-cabecalho {{
    background: linear-gradient(135deg, var(--azul-900), var(--azul-500));
    color: white;
    padding: 28px 32px;
    border-radius: 18px;
    margin-bottom: 24px;
}}
.bloco-cabecalho .selo {{
    display: inline-block;
    font-size: 12px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    background: rgba(255,255,255,0.16);
    padding: 3px 12px;
    border-radius: 999px;
    margin-bottom: 10px;
}}
.bloco-cabecalho h1 {{
    margin: 0 0 6px;
    font-size: 30px;
    font-weight: 700;
}}
.bloco-cabecalho p {{
    margin: 0;
    opacity: 0.92;
    font-size: 15px;
    max-width: 760px;
}}

.cartao-nav {{
    display: block;
    background: white;
    border: 1px solid var(--cinza-300);
    border-radius: 14px;
    padding: 18px 20px;
    height: 100%;
    text-decoration: none;
    transition: box-shadow 0.15s ease, transform 0.15s ease;
}}
.cartao-nav:hover {{
    box-shadow: 0 4px 14px rgba(11, 37, 69, 0.12);
    transform: translateY(-1px);
}}
.cartao-nav .icone {{ font-size: 26px; margin-bottom: 8px; }}
.cartao-nav .titulo {{ font-weight: 700; color: var(--azul-900); font-size: 15px; margin-bottom: 4px; }}
.cartao-nav .desc {{ color: var(--cinza-600); font-size: 13px; line-height: 1.4; }}

.caixa-alerta {{
    border-left: 4px solid var(--ambar-600);
    background: #fff8ec;
    padding: 12px 16px;
    border-radius: 8px;
    margin: 10px 0;
    font-size: 14px;
}}
.caixa-sucesso {{
    border-left: 4px solid var(--verde-600);
    background: #eefaf1;
    padding: 12px 16px;
    border-radius: 8px;
    margin: 10px 0;
    font-size: 14px;
}}
.caixa-info {{
    border-left: 4px solid var(--azul-500);
    background: var(--azul-100);
    padding: 12px 16px;
    border-radius: 8px;
    margin: 10px 0;
    font-size: 14px;
}}

div[data-testid="stMetric"] {{
    background: white;
    border: 1px solid var(--cinza-300);
    border-radius: 12px;
    padding: 14px 16px;
}}

.rodape-app {{
    text-align: center;
    color: var(--cinza-600);
    font-size: 12px;
    padding: 32px 0 8px;
}}
</style>
"""


def aplicar_tema() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def cabecalho(titulo: str, descricao: str, selo: str = "Econometria") -> None:
    st.markdown(
        f"""
        <div class="bloco-cabecalho">
            <span class="selo">{selo}</span>
            <h1>{titulo}</h1>
            <p>{descricao}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def caixa(texto: str, tipo: str = "info") -> None:
    classe = {"info": "caixa-info", "sucesso": "caixa-sucesso", "alerta": "caixa-alerta"}.get(tipo, "caixa-info")
    st.markdown(f'<div class="{classe}">{texto}</div>', unsafe_allow_html=True)


def rodape() -> None:
    st.markdown(
        '<div class="rodape-app">Sistema Econometria &middot; calculos estatisticos, econometricos e '
        "financeiros para uso pessoal e empresarial.</div>",
        unsafe_allow_html=True,
    )
