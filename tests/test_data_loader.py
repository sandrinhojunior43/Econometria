import pandas as pd
import pytest

from econometria.data import clean_dataframe, infer_column_types, load_file, profile_dataset
from econometria.utils import ValidationError


def test_load_file_csv(tmp_path):
    caminho = tmp_path / "dados.csv"
    caminho.write_text("a,b\n1,2\n3,4\n")
    df = load_file(caminho)
    assert list(df.columns) == ["a", "b"]
    assert len(df) == 2


def test_load_file_extensao_invalida(tmp_path):
    caminho = tmp_path / "dados.xyz"
    caminho.write_text("a,b\n1,2\n")
    with pytest.raises(ValidationError):
        load_file(caminho)


def test_load_file_vazio_gera_erro(tmp_path):
    caminho = tmp_path / "vazio.csv"
    caminho.write_text("a,b\n")
    with pytest.raises(ValidationError):
        load_file(caminho)


def test_infer_column_types_numerico_data_categorico():
    df = pd.DataFrame(
        {
            "valor": [1.0, 2.0, 3.0],
            "data": ["2020-01-01", "2020-02-01", "2020-03-01"],
            "categoria": ["a", "b", "a"],
        }
    )
    tipos = infer_column_types(df)
    assert tipos["valor"] == "numerica"
    assert tipos["data"] == "data"
    assert tipos["categoria"] == "categorica"


def test_infer_column_types_numero_formato_brasileiro():
    df = pd.DataFrame({"preco": ["1.234,56", "2.000,00", "999,90"]})
    tipos = infer_column_types(df)
    assert tipos["preco"] == "numerica"


def test_clean_dataframe_converte_decimal_br_e_remove_vazios():
    df = pd.DataFrame({"preco": ["1.234,56", "", "999,90"], "nome": ["a", None, "c"]})
    limpo = clean_dataframe(df)
    assert limpo["preco"].iloc[0] == pytest.approx(1234.56)
    assert len(limpo) == 2  # a linha totalmente vazia nao existia, mas a linha com NaN em preco permanece


def test_profile_dataset_identifica_estrutura_painel():
    df = pd.DataFrame(
        {
            "empresa": ["A", "A", "B", "B"],
            "ano": pd.to_datetime(["2020-01-01", "2021-01-01", "2020-01-01", "2021-01-01"]),
            "receita": [100, 110, 200, 220],
        }
    )
    perfil = profile_dataset(df)
    assert perfil.possivel_painel is True
    assert perfil.coluna_id_painel == "empresa"
