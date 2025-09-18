# -*- coding: utf-8 -*-
# Análise preliminar (versão MEDIANAS)
# - Estatísticas descritivas
# - Agregações por Região, Estado, Município, Bairro (dentro do Estado), Bandeira e Ano
# - Gráficos usando PREÇO MEDIANO (não média)

from pathlib import Path
from typing import Optional, Sequence
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============= Configurações ============
CSV_PATH = "lubricants_anp.csv"  # ajuste se necessário
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

ESTADO_FOCO: Optional[str] = None
TOP_N_ESTADOS = 15
TOP_N_MUNICIPIOS = 20
TOP_N_BAIRROS = 15
TOP_N_BANDEIRAS = 15

# ============= Utilidades de parsing/limpeza ============
def to_float_bra(x) -> float:
    if x is None:
        return np.nan
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return np.nan
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".")
        try:
            return float(s)
        except Exception:
            return np.nan
    if "," in s:
        try:
            return float(s.replace(",", "."))
        except Exception:
            return np.nan
    try:
        return float(s)
    except Exception:
        return np.nan

def read_any_csv(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    except Exception:
        try:
            return pd.read_csv(path, sep=";", encoding="utf-8-sig")
        except Exception:
            return pd.read_csv(path, sep="\t", encoding="utf-8-sig")

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip() for c in df.columns]
    return df

def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    if COL_MAP["data"] in df.columns:
        df[COL_MAP["data"]] = pd.to_datetime(df[COL_MAP["data"]], dayfirst=True, errors="coerce")
        df["Ano"] = df[COL_MAP["data"]].dt.year
        df["Mes"] = df[COL_MAP["data"]].dt.month
        df["AnoMes"] = df[COL_MAP["data"]].dt.to_period("M").astype(str)
    else:
        df["Ano"] = np.nan
        df["Mes"] = np.nan
        df["AnoMes"] = np.nan

    if COL_MAP["preco_venda"] in df.columns:
        df[COL_MAP["preco_venda"]] = df[COL_MAP["preco_venda"]].apply(to_float_bra)
    if COL_MAP["preco_compra"] in df.columns:
        df[COL_MAP["preco_compra"]] = df[COL_MAP["preco_compra"]].apply(to_float_bra)

    for k in ["regiao", "uf", "municipio", "bairro", "produto", "bandeira", "revenda", "unidade", "cep"]:
        col = COL_MAP.get(k)
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    return df

def add_valor_mercadoria_proxy(df: pd.DataFrame) -> pd.DataFrame:
    qty_col = None
    for cand in ["Quantidade", "Quantidade (L)", "Litros", "Volume (L)"]:
        if cand in df.columns:
            qty_col = cand
            break
    if qty_col:
        df["Valor_Mercadoria"] = df[COL_MAP["preco_venda"]] * df[qty_col].apply(to_float_bra)
    else:
        df["Valor_Mercadoria"] = df[COL_MAP["preco_venda"]]
    return df

# ============= Agregações (usando MEDIANA) ============
def agregacoes_basicas(df: pd.DataFrame, keys: Sequence[str]) -> pd.DataFrame:
    g = (
        df.groupby(list(keys), dropna=False)
        .agg(
            n_coletas=(COL_MAP["preco_venda"], "size"),
            preco_mediano=(COL_MAP["preco_venda"], "median"),
            preco_p10=(COL_MAP["preco_venda"], lambda s: np.nanpercentile(s.to_numpy(dtype=float), 10) if len(s) else np.nan),
            preco_p90=(COL_MAP["preco_venda"], lambda s: np.nanpercentile(s.to_numpy(dtype=float), 90) if len(s) else np.nan),
            valor_mercadoria_total=("Valor_Mercadoria", "sum"),
        )
        .reset_index()
        .sort_values(["valor_mercadoria_total", "preco_mediano"], ascending=[False, False])
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
    if not Path(CSV_PATH).exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {CSV_PATH}. Ajuste CSV_PATH.")
    df = read_any_csv(CSV_PATH)
    df = normalize_columns(df)
    df = coerce_types(df)
    df = add_valor_mercadoria_proxy(df)

    print("\n=== Estatísticas gerais do preço de venda ===")
    desc = df[COL_MAP["preco_venda"]].describe(percentiles=[0.1, 0.5, 0.9]).to_frame().T
    print(desc)

    if COL_MAP["produto"] in df.columns:
        print("\n=== Preço MEDIANO por Produto (com dispersão) ===")
        prod_tbl = (
            df.groupby(COL_MAP["produto"])[COL_MAP["preco_venda"]]
            .agg(
                n="size",
                mediana="median",
                p10=lambda s: np.nanpercentile(s.to_numpy(dtype=float), 10) if len(s) else np.nan,
                p90=lambda s: np.nanpercentile(s.to_numpy(dtype=float), 90) if len(s) else np.nan,
                minimo="min",
                maximo="max",
            )
            .sort_values("mediana", ascending=False)
            .round(3)
        )
        print(prod_tbl)

    # 3.1 Região
    if COL_MAP["regiao"] in df.columns:
        agg_regiao = agregacoes_basicas(df, [COL_MAP["regiao"]])
        print("\n=== Agregação por Região (MEDIANA) ===")
        print(agg_regiao.head(20).round(3))
        barplot(
            agg_regiao.sort_values("preco_mediano", ascending=False),
            x_col=COL_MAP["regiao"],
            y_col="preco_mediano",
            title="Preço MEDIANO por Região",
            fname="preco_mediano_por_regiao.png",
            rotate_xticks=0,
        )

    # 3.2 Estado (UF)
    if COL_MAP["uf"] in df.columns:
        agg_uf = agregacoes_basicas(df, [COL_MAP["uf"]])
        print("\n=== Agregação por Estado (UF) — MEDIANA ===")
        print(agg_uf.head(30).round(3))
        top_ufs = agg_uf.sort_values("n_coletas", ascending=False).head(TOP_N_ESTADOS)
        barplot(
            top_ufs.sort_values("preco_mediano", ascending=False),
            x_col=COL_MAP["uf"],
            y_col="preco_mediano",
            title=f"Preço MEDIANO por UF (Top {TOP_N_ESTADOS} em nº de coletas)",
            fname="preco_mediano_por_uf.png",
            rotate_xticks=0,
        )

        # 3.3 Município
        if COL_MAP["municipio"] in df.columns:
            agg_mun = agregacoes_basicas(df, [COL_MAP["uf"], COL_MAP["municipio"]])
            print("\n=== Agregação por Município (UF × Município) — MEDIANA ===")
            print(agg_mun.head(30).round(3))
            top_mun = agg_mun.sort_values("n_coletas", ascending=False).head(TOP_N_MUNICIPIOS)
            top_mun["UF-Município"] = top_mun[COL_MAP["uf"]] + " - " + top_mun[COL_MAP]["municipio"]
            barplot(
                top_mun.sort_values("preco_mediano", ascending=False),
                x_col="UF-Município",
                y_col="preco_mediano",
                title=f"Preço MEDIANO por Município (Top {TOP_N_MUNICIPIOS} em nº de coletas)",
                fname="preco_mediano_por_municipio.png",
                rotate_xticks=60,
            )

        # 3.4 Bairro (no Estado foco)
        if COL_MAP["bairro"] in df.columns:
            estado_foco = ESTADO_FOCO
            if estado_foco is None and df[COL_MAP["uf"]].notna().any():
                estado_foco = (
                    df.groupby(COL_MAP["uf"])[COL_MAP["preco_venda"]]
                    .size()
                    .sort_values(ascending=False)
                    .index[0]
                )
            if estado_foco:
                df_uf = df.loc[df[COL_MAP["uf"]] == estado_foco].copy()
                agg_bairro = agregacoes_basicas(df_uf, [COL_MAP["bairro"]])
                print(f"\n=== Agregação por Bairro no Estado {estado_foco} — MEDIANA ===")
                print(agg_bairro.head(30).round(3))
                top_bairros = agg_bairro.sort_values("n_coletas", ascending=False).head(TOP_N_BAIRROS)
                barplot(
                    top_bairros.sort_values("preco_mediano", ascending=False),
                    x_col=COL_MAP["bairro"],
                    y_col="preco_mediano",
                    title=f"Preço MEDIANO por Bairro no Estado {estado_foco} (Top {TOP_N_BAIRROS})",
                    fname=f"preco_mediano_por_bairro_{estado_foco}.png",
                    rotate_xticks=60,
                )

    # 3.5 Bandeira
    if COL_MAP["bandeira"] in df.columns:
        agg_band = agregacoes_basicas(df, [COL_MAP["bandeira"]])
        print("\n=== Agregação por Bandeira (Distribuidora) — MEDIANA ===")
        print(agg_band.head(30).round(3))
        top_band = agg_band.sort_values("n_coletas", ascending=False).head(TOP_N_BANDEIRAS)
        barplot(
            top_band.sort_values("preco_mediano", ascending=False),
            x_col=COL_MAP["bandeira"],
            y_col="preco_mediano",
            title=f"Preço MEDIANO por Bandeira (Top {TOP_N_BANDEIRAS} em nº de coletas)",
            fname="preco_mediano_por_bandeira.png",
            rotate_xticks=60,
        )

    # 3.6 Ano (série temporal agregada)
    if "Ano" in df.columns and df["Ano"].notna().any():
        agg_ano = agregacoes_basicas(df, ["Ano"]).sort_values("Ano")
        print("\n=== Agregação por Ano — MEDIANA ===")
        print(agg_ano.round(3))
        lineplot(
            agg_ano,
            x_col="Ano",
            y_col="preco_mediano",
            title="Preço MEDIANO por Ano",
            fname="preco_mediano_por_ano.png",
        )
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
