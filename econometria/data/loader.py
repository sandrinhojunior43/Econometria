"""Ingestao, limpeza e perfilamento automatico de dados.

Suporta CSV, Excel (.xlsx/.xls) e JSON, tanto a partir de um caminho em disco
quanto de um objeto tipo-arquivo (por exemplo o retorno de ``st.file_uploader``
do Streamlit). A ideia e que o usuario possa jogar "qualquer tipo de dado" no
sistema e o sistema tente entender o formato sozinho, sinalizando o que nao
conseguiu inferir com confianca.
"""
from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Union

import numpy as np
import pandas as pd

from ..utils.validation import ValidationError

FileLike = Union[str, Path, BinaryIO]


def _read_by_extension(file: FileLike, filename: str, **kwargs) -> pd.DataFrame:
    ext = Path(filename).suffix.lower()
    if ext in (".csv", ".txt"):
        # Tenta separador automatico (virgula ou ponto-e-virgula, comum em dados BR)
        try:
            return pd.read_csv(file, sep=None, engine="python", **kwargs)
        except Exception:
            if hasattr(file, "seek"):
                file.seek(0)
            return pd.read_csv(file, **kwargs)
    if ext in (".xlsx", ".xls", ".xlsm"):
        return pd.read_excel(file, **kwargs)
    if ext == ".json":
        if hasattr(file, "read"):
            payload = json.load(file)
        else:
            payload = json.loads(Path(file).read_text(encoding="utf-8"))
        return pd.json_normalize(payload)
    if ext == ".parquet":
        return pd.read_parquet(file, **kwargs)
    raise ValidationError(
        f"Formato de arquivo '{ext}' nao suportado. Formatos aceitos: CSV, XLSX, XLS, JSON, PARQUET."
    )


def load_file(file: FileLike, filename: str | None = None, **kwargs) -> pd.DataFrame:
    """Carrega um arquivo de dados em um DataFrame, detectando o formato pela extensao.

    Parameters
    ----------
    file:
        Caminho (str/Path) ou objeto tipo-arquivo (ex.: upload do Streamlit).
    filename:
        Nome do arquivo, usado para deduzir a extensao quando ``file`` nao e um path.
    """
    if filename is None:
        filename = getattr(file, "name", None) or str(file)
    try:
        df = _read_by_extension(file, filename, **kwargs)
    except ValidationError:
        raise
    except Exception as exc:  # pragma: no cover - mensagens de erro variam por engine
        raise ValidationError(f"Nao foi possivel ler o arquivo '{filename}': {exc}") from exc

    if df.empty:
        raise ValidationError(f"O arquivo '{filename}' foi lido mas nao contem nenhuma linha de dados.")

    df.columns = [str(c).strip() for c in df.columns]
    return df


def _eh_dtype_texto(serie: pd.Series) -> bool:
    """True para colunas de texto, seja o classico dtype 'object' ou o dtype 'string'
    (StringDtype) que o pandas >=2.x/3.x pode usar por padrao ou via `dtype_backend`.
    """
    return serie.dtype == object or pd.api.types.is_string_dtype(serie)


_BR_DECIMAL_RE = None  # compilado sob demanda para nao pagar custo se nao usado


def _try_parse_brazilian_number(series: pd.Series) -> pd.Series | None:
    """Tenta converter uma coluna de texto no formato '1.234,56' para float."""
    global _BR_DECIMAL_RE
    if _BR_DECIMAL_RE is None:
        import re

        _BR_DECIMAL_RE = re.compile(r"^-?\d{1,3}(\.\d{3})*(,\d+)?$|^-?\d+(,\d+)?$")

    amostra = series.dropna().astype(str).str.strip()
    if amostra.empty:
        return None
    candidatos = amostra.head(200)
    match_rate = candidatos.map(lambda v: bool(_BR_DECIMAL_RE.match(v))).mean()
    if match_rate < 0.9:
        return None
    convertido = series.astype(str).str.strip().str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    return pd.to_numeric(convertido, errors="coerce")


def infer_column_types(df: pd.DataFrame) -> dict[str, str]:
    """Classifica cada coluna em: 'numerica', 'data', 'categorica' ou 'texto'."""
    tipos: dict[str, str] = {}
    for col in df.columns:
        serie = df[col]
        if pd.api.types.is_numeric_dtype(serie):
            tipos[col] = "numerica"
            continue
        if pd.api.types.is_datetime64_any_dtype(serie):
            tipos[col] = "data"
            continue
        if _eh_dtype_texto(serie):
            convertido_num = _try_parse_brazilian_number(serie)
            if convertido_num is not None and convertido_num.notna().mean() > 0.9:
                tipos[col] = "numerica"
                continue
            amostra = serie.dropna().astype(str)
            if not amostra.empty:
                try:
                    parsed = pd.to_datetime(amostra.head(50), errors="coerce", format="mixed")
                    if parsed.notna().mean() > 0.9:
                        tipos[col] = "data"
                        continue
                except Exception:
                    pass
            n_unicos = serie.nunique(dropna=True)
            if n_unicos <= max(20, int(0.2 * len(serie))):
                tipos[col] = "categorica"
            else:
                tipos[col] = "texto"
        else:
            tipos[col] = "texto"
    return tipos


def clean_dataframe(
    df: pd.DataFrame,
    drop_empty_rows: bool = True,
    drop_empty_cols: bool = True,
    strip_strings: bool = True,
    parse_numeric_br: bool = True,
    parse_dates: bool = True,
) -> pd.DataFrame:
    """Limpeza automatica e conservadora: nao inventa dados, apenas padroniza formato."""
    out = df.copy()

    if strip_strings:
        obj_cols = [c for c in out.columns if _eh_dtype_texto(out[c])]
        for c in obj_cols:
            out[c] = out[c].apply(lambda v: v.strip() if isinstance(v, str) else v)
            out[c] = out[c].replace({"": np.nan, "NA": np.nan, "N/A": np.nan, "-": np.nan, "null": np.nan})

    if drop_empty_rows:
        out = out.dropna(how="all")
    if drop_empty_cols:
        out = out.dropna(axis=1, how="all")

    tipos = infer_column_types(out)
    if parse_numeric_br:
        for col, tipo in tipos.items():
            if tipo == "numerica" and _eh_dtype_texto(out[col]):
                convertido = _try_parse_brazilian_number(out[col])
                if convertido is not None:
                    out[col] = convertido
                else:
                    out[col] = pd.to_numeric(out[col], errors="coerce")
    if parse_dates:
        for col, tipo in tipos.items():
            if tipo == "data" and not pd.api.types.is_datetime64_any_dtype(out[col]):
                out[col] = pd.to_datetime(out[col], errors="coerce", format="mixed")

    out = out.reset_index(drop=True)
    return out


@dataclass
class DatasetProfile:
    n_linhas: int
    n_colunas: int
    tipos: dict[str, str]
    faltantes_por_coluna: dict[str, int]
    percentual_faltante_geral: float
    colunas_constantes: list[str]
    colunas_data: list[str] = field(default_factory=list)
    colunas_numericas: list[str] = field(default_factory=list)
    colunas_categoricas: list[str] = field(default_factory=list)
    possivel_serie_temporal: bool = False
    possivel_painel: bool = False
    coluna_id_painel: str | None = None
    coluna_tempo_painel: str | None = None
    resumo_numerico: pd.DataFrame | None = None

    def to_markdown_bullets(self) -> str:
        linhas = [
            f"- **{self.n_linhas}** linhas x **{self.n_colunas}** colunas.",
            f"- Dados faltantes: **{self.percentual_faltante_geral:.1f}%** do total de celulas.",
        ]
        if self.colunas_constantes:
            linhas.append(f"- Colunas constantes (sem variacao): {', '.join(self.colunas_constantes)}.")
        if self.possivel_serie_temporal:
            linhas.append("- Estrutura compativel com **serie temporal** (ha coluna de data/tempo).")
        if self.possivel_painel:
            linhas.append(
                f"- Estrutura compativel com **dados em painel** "
                f"(identificador: `{self.coluna_id_painel}`, tempo: `{self.coluna_tempo_painel}`)."
            )
        return "\n".join(linhas)


def profile_dataset(df: pd.DataFrame) -> DatasetProfile:
    """Gera um raio-x automatico do dataset para orientar qual analise faz sentido."""
    tipos = infer_column_types(df)
    faltantes = df.isna().sum().to_dict()
    pct_faltante = float(df.isna().mean().mean() * 100) if df.size else 0.0

    colunas_constantes = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
    colunas_data = [c for c, t in tipos.items() if t == "data"]
    colunas_numericas = [c for c, t in tipos.items() if t == "numerica"]
    colunas_categoricas = [c for c, t in tipos.items() if t == "categorica"]

    possivel_serie_temporal = len(colunas_data) >= 1 and len(colunas_numericas) >= 1

    # Heuristica simples de painel: uma coluna categorica/id com poucos valores unicos
    # repetidos ao longo de uma coluna de data -> parece id x tempo.
    possivel_painel = False
    col_id_painel = None
    col_tempo_painel = None
    if colunas_data:
        col_tempo_painel = colunas_data[0]
        for c in colunas_categoricas:
            n_unicos = df[c].nunique(dropna=True)
            if 1 < n_unicos < len(df):
                media_obs_por_id = len(df) / n_unicos
                if media_obs_por_id >= 2:
                    possivel_painel = True
                    col_id_painel = c
                    break

    resumo_numerico = df[colunas_numericas].describe().T if colunas_numericas else None

    return DatasetProfile(
        n_linhas=len(df),
        n_colunas=len(df.columns),
        tipos=tipos,
        faltantes_por_coluna={k: int(v) for k, v in faltantes.items()},
        percentual_faltante_geral=pct_faltante,
        colunas_constantes=colunas_constantes,
        colunas_data=colunas_data,
        colunas_numericas=colunas_numericas,
        colunas_categoricas=colunas_categoricas,
        possivel_serie_temporal=possivel_serie_temporal,
        possivel_painel=possivel_painel,
        coluna_id_painel=col_id_painel,
        coluna_tempo_painel=col_tempo_painel,
        resumo_numerico=resumo_numerico,
    )
