"""Gerenciamento do dataset ativo (session_state) e widget de upload/exemplos, compartilhados
por todas as paginas do app.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

import _bootstrap  # noqa: F401
from econometria.data import clean_dataframe, load_file, profile_dataset

_EXEMPLOS_DIR = Path(__file__).resolve().parent.parent / "examples"

EXEMPLOS = {
    "Vendas mensais (serie temporal)": "vendas_mensais.csv",
    "Precos de imoveis (regressao)": "imoveis.csv",
    "Churn de clientes (escolha discreta)": "clientes_churn.csv",
    "Desempenho de empresas (painel)": "painel_empresas.csv",
}


def get_dataset() -> pd.DataFrame | None:
    return st.session_state.get("dataset")


def get_dataset_nome() -> str | None:
    return st.session_state.get("dataset_nome")


def set_dataset(df: pd.DataFrame, nome: str) -> None:
    st.session_state["dataset"] = df
    st.session_state["dataset_nome"] = nome


def sidebar_dados() -> None:
    """Widget de sidebar para carregar dados: upload proprio ou dataset de exemplo.
    Compartilhado entre paginas via st.session_state - carregue uma vez, use em qualquer pagina.
    """
    with st.sidebar:
        st.markdown("### 📁 Dados")
        arquivo = st.file_uploader(
            "Envie seus dados (CSV, Excel, JSON)",
            type=["csv", "xlsx", "xls", "json", "parquet"],
            key="uploader_dados",
        )
        if arquivo is not None:
            try:
                bruto = load_file(arquivo, filename=arquivo.name)
                limpo = clean_dataframe(bruto)
                set_dataset(limpo, arquivo.name)
                st.success(f"'{arquivo.name}' carregado: {limpo.shape[0]} linhas x {limpo.shape[1]} colunas.")
            except Exception as exc:
                st.error(f"Nao foi possivel carregar o arquivo: {exc}")

        st.caption("ou experimente com dados de exemplo:")
        escolha = st.selectbox("Dataset de exemplo", ["-"] + list(EXEMPLOS.keys()), key="select_exemplo")
        if escolha != "-" and st.button("Carregar exemplo", width="stretch"):
            caminho = _EXEMPLOS_DIR / EXEMPLOS[escolha]
            bruto = load_file(caminho)
            limpo = clean_dataframe(bruto)
            set_dataset(limpo, EXEMPLOS[escolha])
            st.success(f"Exemplo '{escolha}' carregado.")
            st.rerun()

        df = get_dataset()
        if df is not None:
            st.divider()
            st.caption(f"**Ativo:** {get_dataset_nome()}")
            st.caption(f"{df.shape[0]} linhas x {df.shape[1]} colunas")
            if st.button("🗑️ Limpar dataset", width="stretch"):
                st.session_state.pop("dataset", None)
                st.session_state.pop("dataset_nome", None)
                st.rerun()


def exigir_dataset() -> pd.DataFrame:
    """Interrompe a renderizacao da pagina com uma mensagem amigavel se nao houver dataset ativo."""
    df = get_dataset()
    if df is None:
        st.info(
            "📂 Nenhum dataset carregado ainda. Use a barra lateral para enviar um arquivo "
            "(CSV, Excel ou JSON) ou carregar um dos datasets de exemplo."
        )
        st.stop()
    return df


def mostrar_perfil_dataset(df: pd.DataFrame) -> None:
    perfil = profile_dataset(df)
    with st.expander("🔍 Raio-x automatico dos dados", expanded=False):
        st.markdown(perfil.to_markdown_bullets())
        st.dataframe(df.head(20), width="stretch")
        if perfil.resumo_numerico is not None:
            st.caption("Resumo das colunas numericas")
            st.dataframe(perfil.resumo_numerico, width="stretch")
