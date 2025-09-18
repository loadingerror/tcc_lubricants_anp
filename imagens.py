# -*- coding: utf-8 -*-
# Análise preliminar da base de preços de combustíveis (ANP)
# - Estatísticas descritivas
# - Agregações por Região, Estado, Município, Bairro (dentro do Estado), Bandeira e Ano
# - Gráficos (matplotlib)
#
# OBS:
# 1) "Valor de Compra" é ignorado neste diagnóstico (corte de série).
# 2) Sem coluna de quantidade, tratamos “mercadoria circulada” como preço médio (proxy).
#    Se você tiver uma coluna de quantidade (ex.: 'Quantidade' em litros), ative o bloco marcado.

import os
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============= Configurações ============
CSV_PATH = "lubricants_anp.csv"  
OUTDIR = Path("figs_anp")
OUTDIR.mkdir(parents=True, exist_ok=True)

COL_MAP = {
    "regiao": "Regiao - Sigla",
    "uf": "Estado - Sigla",
    "municipio": "Municipio",
    "revenda": "Revenda",
    "cnpj": "CNPJ da Revenda",
    "rua": "Nome da Rua",
    "numero": "Numero Rua",
    "complemento": "Complemento",
    "bairro": "Bairro",
    "cep": "Cep",
    "produto": "Produto",
    "data": "Data da Coleta",
    "preco_venda": "Valor de Venda",
    "preco_compra": "Valor de Compra",
    "unidade": "Unidade de Medida",
    "bandeira": "Bandeira",
}

# Caso você queira focar um estado específico para o gráfico por Bairros:
ESTADO_FOCO: Optional[str] = None  # ex.: "SC" ou "SP"; se None, escolhe o de maior nº de coletas

# Limites de barras para gráficos (para não poluir a visualização)
TOP_N_ESTADOS = 15
TOP_N_MUNICIPIOS = 20
TOP_N_BAIRROS = 15
TOP_N_BANDEIRAS = 15


# ============= Utilidades de parsing/limpeza ============
def to_float_bra(x) -> float:
    """Converte string numérica BR/US para float (aceita '6,55', '5.897', '1.234,56')."""
    if x is None:
        return np.nan
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return np.nan
    # Se tem '.' e ',' assume formato BR (ponto milhar, vírgula decimal)
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".")
        try:
            return float(s)
        except Exception:
            return np.nan
    # Se só tem ',', assume vírgula decimal
    if "," in s:
        try:
            return float(s.replace(",", "."))
        except Exception:
            return np.nan
    # Caso padrão (US)
    try:
        return float(s)
    except Exception:
        return np.nan


def read_any_csv(path: str) -> pd.DataFrame:
    """Lê CSV/TSV tentando inferir separador e codificação. Ajuste se conhecer o formato exato."""
    # Tenta com inferência de separador (engine=python)
    try:
        df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
        return df
    except Exception:
        # Tenta com ; (muito comum em dados BR)
        try:
            df = pd.read_csv(path, sep=";", encoding="utf-8-sig")
            return df
        except Exception:
            # Tenta com tab
            df = pd.read_csv(path, sep="\t", encoding="utf-8-sig")
            return df


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Padroniza nomes das colunas conforme esperado (mantém original quando não encontrado)."""
    # Remove espaços extras nas colunas
    df.columns = [c.strip() for c in df.columns]
    return df


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """Converte tipos para análise: data, floats, strings."""
    # Datas (dd/mm/aaaa)
    if COL_MAP["data"] in df.columns:
        df[COL_MAP["data"]] = pd.to_datetime(df[COL_MAP["data"]], dayfirst=True, errors="coerce")
        df["Ano"] = df[COL_MAP["data"]].dt.year
        df["Mes"] = df[COL_MAP["data"]].dt.month
        df["AnoMes"] = df[COL_MAP["data"]].dt.to_period("M").astype(str)
    else:
        df["Ano"] = np.nan
        df["Mes"] = np.nan
        df["AnoMes"] = np.nan

    # Preço de venda
    if COL_MAP["preco_venda"] in df.columns:
        df[COL_MAP["preco_venda"]] = df[COL_MAP["preco_venda"]].apply(to_float_bra)

    # Preço de compra (não usado nesta análise, mas já converte)
    if COL_MAP["preco_compra"] in df.columns:
        df[COL_MAP["preco_compra"]] = df[COL_MAP["preco_compra"]].apply(to_float_bra)

    # Alguns campos categóricos como string consistente
    for k in ["regiao", "uf", "municipio", "bairro", "produto", "bandeira", "revenda", "unidade", "cep"]:
        col = COL_MAP.get(k)
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    return df


def add_valor_mercadoria_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria coluna 'Valor_Mercadoria':
    - Se existir 'Quantidade' (litros), usa: Valor de Venda * Quantidade.
    - Caso contrário, usa o próprio 'Valor de Venda' (proxy de nível de preço).
    """
    qty_col = None
    for cand in ["Quantidade", "Quantidade (L)", "Litros", "Volume (L)"]:
        if cand in df.columns:
            qty_col = cand
            break

    if qty_col:
        df["Valor_Mercadoria"] = df[COL_MAP["preco_venda"]] * df[qty_col].apply(to_float_bra)
    else:
        df["Valor_Mercadoria"] = df[COL_MAP["preco_venda"]]  # proxy (sem quantidade)

    return df


# ============= Agregações ============
def agregacoes_basicas(df: pd.DataFrame, keys: Sequence[str]) -> pd.DataFrame:
    """
    Calcula métricas básicas por 'keys':
    - n_coletas
    - preco_medio, preco_mediano, preco_p10, preco_p90
    - valor_mercadoria_total (soma do proxy)
    """
    g = (
        df.groupby(list(keys), dropna=False)
        .agg(
            n_coletas=(COL_MAP["preco_venda"], "size"),
            preco_medio=(COL_MAP["preco_venda"], "mean"),
            preco_mediano=(COL_MAP["preco_venda"], "median"),
            preco_p10=(COL_MAP["preco_venda"], lambda s: np.nanpercentile(s.to_numpy(dtype=float), 10) if len(s) else np.nan),
            preco_p90=(COL_MAP["preco_venda"], lambda s: np.nanpercentile(s.to_numpy(dtype=float), 90) if len(s) else np.nan),
            valor_mercadoria_total=("Valor_Mercadoria", "sum"),
        )
        .reset_index()
        .sort_values(["valor_mercadoria_total", "preco_medio"], ascending=[False, False])
    )
    return g


# ============= Plot helpers ============
def barplot(df: pd.DataFrame, x_col: str, y_col: str, title: str, fname: Optional[str] = None, rotate_xticks: int = 0):
    plt.figure(figsize=(10, 5))
    plt.bar(df[x_col].astype(str), df[y_col].astype(float))
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    if rotate_xticks:
        plt.xticks(rotation=rotate_xticks, ha="right")
    plt.tight_layout()
    if fname:
        plt.savefig(OUTDIR / fname, dpi=120)
    plt.show()


def lineplot(df: pd.DataFrame, x_col: str, y_col: str, title: str, fname: Optional[str] = None):
    plt.figure(figsize=(10, 5))
    plt.plot(df[x_col].astype(str), df[y_col].astype(float), marker="o")
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.tight_layout()
    if fname:
        plt.savefig(OUTDIR / fname, dpi=120)
    plt.show()


# ============= Pipeline principal ============
def main():
    # 1) Carregar dados
    if not Path(CSV_PATH).exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {CSV_PATH}. Ajuste a variável CSV_PATH para apontar ao seu dataset."
        )
    df = read_any_csv(CSV_PATH)
    df = normalize_columns(df)
    df = coerce_types(df)
    df = add_valor_mercadoria_proxy(df)

    # 2) Estatísticas gerais
    print("\n=== Estatísticas gerais do preço de venda ===")
    desc = df[COL_MAP["preco_venda"]].describe(percentiles=[0.1, 0.5, 0.9]).to_frame().T
    print(desc)

    if COL_MAP["produto"] in df.columns:
        print("\n=== Preço médio por Produto ===")
        print(
            df.groupby(COL_MAP["produto"])[COL_MAP["preco_venda"]]
            .agg(["count", "mean", "median", "min", "max"])
            .sort_values("mean", ascending=False)
            .round(3)
        )

    # 3) Agregações por Região, Estado, Município, Bairro (no Estado), Bandeira e Ano
    # 3.1 Região
    if COL_MAP["regiao"] in df.columns:
        agg_regiao = agregacoes_basicas(df, [COL_MAP["regiao"]])
        print("\n=== Agregação por Região ===")
        print(agg_regiao.head(20).round(3))

        # Gráfico: preço médio por Região
        barplot(
            agg_regiao.sort_values("preco_medio", ascending=False),
            x_col=COL_MAP["regiao"],
            y_col="preco_medio",
            title="Preço médio por Região",
            fname="preco_medio_por_regiao.png",
            rotate_xticks=0,
        )

    # 3.2 Estado (UF)
    if COL_MAP["uf"] in df.columns:
        agg_uf = agregacoes_basicas(df, [COL_MAP["uf"]])
        print("\n=== Agregação por Estado (UF) ===")
        print(agg_uf.head(30).round(3))

        # Gráfico: preço médio por UF (Top N por nº de coletas)
        top_ufs = agg_uf.sort_values("n_coletas", ascending=False).head(TOP_N_ESTADOS)
        barplot(
            top_ufs.sort_values("preco_medio", ascending=False),
            x_col=COL_MAP["uf"],
            y_col="preco_medio",
            title=f"Preço médio por UF (Top {TOP_N_ESTADOS} em nº de coletas)",
            fname="preco_medio_por_uf.png",
            rotate_xticks=0,
        )

        # 3.3 Município (dentro do Brasil)
        if COL_MAP["municipio"] in df.columns:
            agg_mun = agregacoes_basicas(df, [COL_MAP["uf"], COL_MAP["municipio"]])
            print("\n=== Agregação por Município (UF × Município) ===")
            print(agg_mun.head(30).round(3))

            # Gráfico: preço médio por Município (Top N por nº de coletas)
            top_mun = agg_mun.sort_values("n_coletas", ascending=False).head(TOP_N_MUNICIPIOS)
            top_mun["UF-Município"] = top_mun[COL_MAP["uf"]] + " - " + top_mun[COL_MAP["municipio"]]
            barplot(
                top_mun.sort_values("preco_medio", ascending=False),
                x_col="UF-Município",
                y_col="preco_medio",
                title=f"Preço médio por Município (Top {TOP_N_MUNICIPIOS} em nº de coletas)",
                fname="preco_medio_por_municipio.png",
                rotate_xticks=60,
            )

        # 3.4 Bairro (dentro do Estado)
        if COL_MAP["bairro"] in df.columns:
            # Escolhe estado foco automaticamente (mais observações) se não definido
            estado_foco = ESTADO_FOCO
            if estado_foco is None:
                estado_foco = (
                    df.groupby(COL_MAP["uf"])[COL_MAP["preco_venda"]]
                    .size()
                    .sort_values(ascending=False)
                    .index[0]
                    if df[COL_MAP["uf"]].notna().any()
                    else None
                )

            if estado_foco:
                df_uf = df.loc[df[COL_MAP["uf"]] == estado_foco].copy()
                agg_bairro = agregacoes_basicas(df_uf, [COL_MAP["bairro"]])
                print(f"\n=== Agregação por Bairro no Estado {estado_foco} ===")
                print(agg_bairro.head(30).round(3))

                top_bairros = agg_bairro.sort_values("n_coletas", ascending=False).head(TOP_N_BAIRROS)
                barplot(
                    top_bairros.sort_values("preco_medio", ascending=False),
                    x_col=COL_MAP["bairro"],
                    y_col="preco_medio",
                    title=f"Preço médio por Bairro no Estado {estado_foco} (Top {TOP_N_BAIRROS} em nº de coletas)",
                    fname=f"preco_medio_por_bairro_{estado_foco}.png",
                    rotate_xticks=60,
                )

    # 3.5 Bandeira (distribuidora)
    if COL_MAP["bandeira"] in df.columns:
        agg_band = agregacoes_basicas(df, [COL_MAP["bandeira"]])
        print("\n=== Agregação por Bandeira (Distribuidora) ===")
        print(agg_band.head(30).round(3))

        top_band = agg_band.sort_values("n_coletas", ascending=False).head(TOP_N_BANDEIRAS)
        barplot(
            top_band.sort_values("preco_medio", ascending=False),
            x_col=COL_MAP["bandeira"],
            y_col="preco_medio",
            title=f"Preço médio por Bandeira (Top {TOP_N_BANDEIRAS} em nº de coletas)",
            fname="preco_medio_por_bandeira.png",
            rotate_xticks=60,
        )

    # 3.6 Ano (série temporal agregada)
    if "Ano" in df.columns and df["Ano"].notna().any():
        agg_ano = agregacoes_basicas(df, ["Ano"]).sort_values("Ano")
        print("\n=== Agregação por Ano ===")
        print(agg_ano.round(3))

        # Gráfico: preço médio por Ano
        lineplot(
            agg_ano,
            x_col="Ano",
            y_col="preco_medio",
            title="Preço médio por Ano",
            fname="preco_medio_por_ano.png",
        )

        # Gráfico: (se houver quantidade no futuro) valor_mercadoria_total por Ano
        lineplot(
            agg_ano,
            x_col="Ano",
            y_col="valor_mercadoria_total",
            title="(Proxy) Valor de mercadoria por Ano",
            fname="valor_mercadoria_por_ano.png",
        )

    print(f"\nArquivos de figuras (PNG) salvos em: {OUTDIR.resolve()}")


if __name__ == "__main__":
    main()
