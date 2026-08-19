"""Componentes de UI reutilizados entre paginas: KPIs, downloads de relatorio, graficos padrao."""
from __future__ import annotations

from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import _bootstrap  # noqa: F401
from econometria.reports import ReportBuilder

PALETA = ["#2563eb", "#0f9d58", "#b7791f", "#d93025", "#7c3aed", "#0891b2"]

PLOTLY_LAYOUT = dict(
    template="plotly_white",
    font=dict(family="Segoe UI, Arial, sans-serif", size=13, color="#1a2233"),
    colorway=PALETA,
    margin=dict(l=10, r=10, t=40, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
)


def estilizar(fig: go.Figure, titulo: str = "") -> go.Figure:
    fig.update_layout(**PLOTLY_LAYOUT, title=titulo)
    return fig


def kpis(itens: list[tuple[str, str, str]]) -> None:
    """itens: lista de (rotulo, valor, ajuda-opcional)."""
    cols = st.columns(len(itens))
    for col, (rotulo, valor, ajuda) in zip(cols, itens):
        col.metric(rotulo, valor, help=ajuda or None)


def botoes_download_relatorio(builder: ReportBuilder, nome_base: str) -> None:
    """Renderiza tres botoes de download (HTML, DOCX, XLSX) para um ReportBuilder pronto."""
    st.markdown("#### 📄 Exportar relatorio")
    carimbo = datetime.now().strftime("%Y%m%d_%H%M")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.download_button(
            "⬇️ HTML (visual)",
            data=builder.render_html().encode("utf-8"),
            file_name=f"{nome_base}_{carimbo}.html",
            mime="text/html",
            width="stretch",
        )
    with col2:
        try:
            st.download_button(
                "⬇️ Word (.docx)",
                data=builder.to_docx_bytes(),
                file_name=f"{nome_base}_{carimbo}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                width="stretch",
            )
        except Exception as exc:
            st.caption(f"Word indisponivel: {exc}")
    with col3:
        try:
            st.download_button(
                "⬇️ Excel (.xlsx)",
                data=builder.to_excel_bytes(),
                file_name=f"{nome_base}_{carimbo}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                width="stretch",
            )
        except Exception as exc:
            st.caption(f"Excel indisponivel: {exc}")


def selecionar_colunas_numericas(df: pd.DataFrame, rotulo: str, padrao: list[str] | None = None, **kwargs) -> list[str]:
    numericas = df.select_dtypes(include="number").columns.tolist()
    return st.multiselect(rotulo, numericas, default=padrao or [], **kwargs)
