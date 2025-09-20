import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# 0) Configurações gerais
# =========================
ARQUIVO = "lubricants_anp_full.csv"        # << ajuste o nome do arquivo
PASTA_OUT_ROOT = "_out_anual"
os.makedirs(PASTA_OUT_ROOT, exist_ok=True)

# Tamanho padrão dos gráficos
plt.rcParams["figure.figsize"] = (9, 5)
plt.rcParams["axes.grid"] = True

# =========================
# 1) Utilidades de leitura
# =========================
def carregar_csv_curinga(path):
    seps = [",", ";", "\t", "|"]
    encodings = ["utf-8", "latin-1"]
    ultimo_erro = None
    for enc in encodings:
        for sep in seps:
            try:
                df = pd.read_csv(path, sep=sep, encoding=enc, engine="python", dtype=str)
                if df.shape[1] == 1:  # provavelmente separador errado
                    continue
                return df
            except Exception as e:
                ultimo_erro = e
                continue
    raise RuntimeError(f"Falha ao ler {path}. Último erro: {ultimo_erro}")

raw = carregar_csv_curinga(ARQUIVO)

# =========================
# 2) Padronização de colunas
# =========================
def normalizar_nome(col):
    col = col.strip().lower()
    col = col.replace(" ", "_").replace("-", "_").replace(".", "_")
    col = re.sub(r"__+", "_", col)
    return col

raw.columns = [normalizar_nome(c) for c in raw.columns]

aliases = {
    "regiao": "regiao_sigla", "regiao_sigla": "regiao_sigla", "sigla_regiao": "regiao_sigla",
    "estado": "estado_sigla", "uf": "estado_sigla", "estado_sigla": "estado_sigla",
    "municipio": "municipio",
    "produto": "produto", "produto_nome": "produto",
    "data_coleta": "data_coleta", "data": "data_coleta", "dt_coleta": "data_coleta",
    "valor_venda": "valor_venda", "preco_venda": "valor_venda", "preco": "valor_venda",
    "unidade_medida": "unidade_medida",
    "bandeira": "bandeira",
    "ano": "ano",
}

ren = {}
for c in raw.columns:
    if c in aliases:
        ren[c] = aliases[c]
raw = raw.rename(columns=ren)

# =========================
# 3) Limpeza e tipos
# =========================
df = raw.copy()
for c in ["produto", "bandeira", "regiao_sigla", "estado_sigla", "municipio"]:
    if c not in df.columns:
        df[c] = np.nan
    df[c] = df[c].astype(str).str.strip().str.upper()

if "data_coleta" not in df.columns:
    df["data_coleta"] = np.nan

def parse_data(x):
    x = str(x).strip()
    if x in ("", "nan", "na", "none"):
        return pd.NaT
    # tenta formatos mais comuns; depois inferência
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%Y/%m/%d", "%d-%m-%Y"):
        try:
            return pd.to_datetime(x, format=fmt)
        except Exception:
            pass
    return pd.to_datetime(x, dayfirst=True, errors="coerce")

df["data_coleta"] = df["data_coleta"].apply(parse_data)

def to_int_or_nan(x):
    try:
        return int(float(str(x).strip()))
    except Exception:
        return np.nan

if "ano" not in df.columns:
    df["ano"] = np.nan
else:
    df["ano"] = df["ano"].apply(to_int_or_nan)

# Preenche ano a partir da data, se faltar
mask_ano_na = df["ano"].isna() & df["data_coleta"].notna()
df.loc[mask_ano_na, "ano"] = df.loc[mask_ano_na, "data_coleta"].dt.year

# Valor de venda -> numérico
def parse_valor(s):
    if pd.isna(s):
        return np.nan
    s = str(s)
    s = s.replace("R$", "").replace("r$", "")
    s = s.replace("/ litro", "").replace("/litro", "")
    # remove espaços e caracteres não numéricos (mantém . e -)
    s = re.sub(r"[^0-9\.\-,]", "", s).strip()
    # Trata casos com ponto de milhar e vírgula decimal
    # Se há uma vírgula e múltiplos pontos, remove pontos de milhar e troca vírgula por ponto
    if s.count(",") == 1 and s.count(".") > 1:
        s = s.replace(".", "").replace(",", ".")
    else:
        s = s.replace(",", ".")
    try:
        return float(s)
    except Exception:
        return np.nan

if "valor_venda" not in df.columns:
    raise ValueError("Coluna 'valor_venda' ausente.")
df["valor_venda"] = df["valor_venda"].apply(parse_valor)

# Mes (YYYY-MM) — útil para mediana mensal
df["mes"] = np.where(df["data_coleta"].notna(), df["data_coleta"].dt.to_period("M").astype(str), np.nan)

# Remove registros inválidos
df = df[~df["ano"].isna() & ~df["valor_venda"].isna()].copy()
df["ano"] = df["ano"].astype(int)

# =========================
# 4) Funções de análise
# =========================
def stats_descritivas(data, group_cols=None, min_obs=1):
    """
    Calcula n, média, mediana, std, min, max e CV para valor_venda.
    Se group_cols for None -> agrega geral.
    """
    by = []
    if group_cols:
        by = group_cols if isinstance(group_cols, list) else [group_cols]
    g = (data
         .groupby(by)["valor_venda"]
         .agg(n="count", media="mean", mediana="median", desvio_padrao="std",
              minimo="min", maximo="max")
         .reset_index())
    # CV
    g["cv"] = g["desvio_padrao"] / g["media"]
    # Filtro opcional por contagem
    g = g[g["n"] >= min_obs].copy()
    return g

def grafico_mediana_mensal(data, titulo, caminho_png):
    """
    Gera gráfico da mediana mensal de valor_venda (um gráfico por figura).
    """
    d = data.dropna(subset=["data_coleta"]).copy()
    if d.empty:
        return False

    # Série mensal completa do ano (jan..dez)
    ano = int(d["ano"].iloc[0])
    s = (d.set_index("data_coleta")
           .resample("M")["valor_venda"].median())
    # Garante meses vazios como NaN (jan..dez)
    idx = pd.period_range(f"{ano}-01", f"{ano}-12", freq="M").to_timestamp()
    s = s.reindex(idx)

    plt.figure()
    plt.plot(s.index.strftime("%Y-%m"), s.values, marker="o")
    plt.title(titulo)
    plt.xlabel("Mês")
    plt.ylabel("Mediana mensal do valor_venda")
    # Anota valores
    for x, y in zip(range(len(s.index)), s.values):
        if pd.notna(y):
            plt.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0,6), ha="center")
    plt.xticks(range(len(s.index)), s.index.strftime("%b"), rotation=0)
    plt.tight_layout()
    plt.savefig(caminho_png, dpi=150)
    plt.close()
    return True

def salvar_csv(df_, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df_.to_csv(path, index=False)

def executar_bloco_anual(df_base, ano, notas_eventos=None):
    """
    Executa TODA a análise para um ano específico.
    - Cria pasta ./_out_anual/{ano}
    - Salva CSVs de estatísticas (geral, produto, bandeira, regiao, estado, municipio)
    - Gera gráfico de mediana mensal (geral do ano)
    - Escreve um README_ano.txt com notas de eventos (se fornecido)
    """
    sub = df_base[df_base["ano"] == ano].copy()
    outdir = os.path.join(PASTA_OUT_ROOT, str(ano))
    os.makedirs(outdir, exist_ok=True)

    if sub.empty:
        with open(os.path.join(outdir, f"README_{ano}.txt"), "w", encoding="utf-8") as f:
            f.write(f"Nenhum dado disponível para {ano}.\n")
        print(f"[{ano}] Sem dados.")
        return

    # --- Tabelas descritivas (por ano) ---
    geral       = stats_descritivas(sub, None)
    por_prod    = stats_descritivas(sub, "produto")
    por_band    = stats_descritivas(sub, "bandeira")
    por_regiao  = stats_descritivas(sub, "regiao_sigla")
    por_uf      = stats_descritivas(sub, "estado_sigla")
    por_mun     = stats_descritivas(sub, ["estado_sigla", "municipio"])

    salvar_csv(geral,      os.path.join(outdir, f"{ano}_stats_geral.csv"))
    salvar_csv(por_prod,   os.path.join(outdir, f"{ano}_stats_produto.csv"))
    salvar_csv(por_band,   os.path.join(outdir, f"{ano}_stats_bandeira.csv"))
    salvar_csv(por_regiao, os.path.join(outdir, f"{ano}_stats_regiao.csv"))
    salvar_csv(por_uf,     os.path.join(outdir, f"{ano}_stats_uf.csv"))
    salvar_csv(por_mun,    os.path.join(outdir, f"{ano}_stats_municipio.csv"))

    # --- Gráfico: Mediana mensal (geral do ano) ---
    grafico_mediana_mensal(
        sub,
        titulo=f"Mediana mensal do valor_venda – {ano} (Geral)",
        caminho_png=os.path.join(outdir, f"{ano}_mediana_mensal_geral.png")
    )

    # --- README com notas de eventos/políticas (para você preencher) ---
    with open(os.path.join(outdir, f"README_{ano}.txt"), "w", encoding="utf-8") as f:
        f.write(f"Resumo {ano}\n")
        f.write("-" * 40 + "\n")
        if notas_eventos:
            if isinstance(notas_eventos, (list, tuple)):
                for linha in notas_eventos:
                    f.write(f"- {linha}\n")
            else:
                f.write(f"- {notas_eventos}\n")
        else:
            f.write("(Adicione aqui suas notas de políticas/mercado – ICMS, câmbio, Brent, etc.)\n")

    print(f"[{ano}] Concluído. Resultados em: {outdir}")

# =========================
# 5) NOTAS (edite à vontade)
# =========================
NOTAS = {
    2017: ["(exemplo) Reoneração parcial de tributos; dinâmica do etanol vs. açúcar"],
    2018: ["(exemplo) Greve dos caminhoneiros em maio/2018 impactando diesel"],
    2019: [],
    2020: ["(exemplo) Pandemia: queda de demanda; volatilidade cambial"],
    2021: ["(exemplo) Recuperação da mobilidade; pressões de custo no diesel"],
    2022: ["(exemplo) Mudanças em ICMS/tributos; choque internacional"],
    2023: ["(exemplo) Ajustes tributários; política de preços e câmbio"],
    2024: [],
    2025: [],
}

# =========================
# 6) BLOCOS ANUAIS (2017 → 2025)
# =========================
# Observação: se algum ano não existir na base, o bloco cria a pasta e um README informando "Sem dados".

# -------- 2017 --------
executar_bloco_anual(df, 2017, NOTAS.get(2017))

# -------- 2018 --------
executar_bloco_anual(df, 2018, NOTAS.get(2018))

# -------- 2019 --------
executar_bloco_anual(df, 2019, NOTAS.get(2019))

# -------- 2020 --------
executar_bloco_anual(df, 2020, NOTAS.get(2020))

# -------- 2021 --------
executar_bloco_anual(df, 2021, NOTAS.get(2021))

# -------- 2022 --------
executar_bloco_anual(df, 2022, NOTAS.get(2022))

# -------- 2023 --------
executar_bloco_anual(df, 2023, NOTAS.get(2023))

# -------- 2024 --------
executar_bloco_anual(df, 2024, NOTAS.get(2024))

# -------- 2025 --------
executar_bloco_anual(df, 2025, NOTAS.get(2025))

print("Pipeline anual finalizado.")
