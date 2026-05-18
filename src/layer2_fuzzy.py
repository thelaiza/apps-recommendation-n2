"""
Camada II — Inferência e Tratamento de Incerteza (Fuzzy Mamdani)
----------------------------------------------------------------
Recebe a probabilidade de sentimento positivo (Camada I) e o rating
médio do app (Google Play) como entradas, e produz um Score de
Atratividade [0–10] via sistema Mamdani.

Entradas:
  - sentimento    : prob. de sentimento positivo em [0, 1]
  - rating        : avaliação média do app normalizada em [0, 1]
                    (original: 1–5, normalizado: (rating-1)/4)

Saída:
  - score_fuzzy   : atratividade do app em [0, 10]

Resultado: results/catalog_fuzzy.csv
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from skfuzzy import control as ctrl

warnings.filterwarnings("ignore")

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
DATA_DIR    = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Universos de discurso

u_sentimento = np.arange(0.0, 1.01, 0.01)
u_rating     = np.arange(0.0, 1.01, 0.01)
u_score      = np.arange(0.0, 10.01, 0.01)

# Variáveis fuzzy

sentimento = ctrl.Antecedent(u_sentimento, "sentimento")
rating     = ctrl.Antecedent(u_rating,     "rating")
score      = ctrl.Consequent(u_score,      "score")

# Funções de pertinência — Sentimento

sentimento["Negativo"] = fuzz.trapmf(u_sentimento, [0.00, 0.00, 0.25, 0.45])
sentimento["Neutro"]   = fuzz.trimf (u_sentimento, [0.35, 0.50, 0.65])
sentimento["Positivo"] = fuzz.trapmf(u_sentimento, [0.55, 0.75, 1.00, 1.00])

# Funções de pertinência — Rating normalizado

rating["Baixo"]  = fuzz.trapmf(u_rating, [0.00, 0.00, 0.25, 0.45])
rating["Medio"]  = fuzz.trimf (u_rating, [0.35, 0.50, 0.65])
rating["Alto"]   = fuzz.trapmf(u_rating, [0.55, 0.75, 1.00, 1.00])

# Funções de pertinência - Score de Atratividade

score["Muito Baixo"] = fuzz.trapmf(u_score, [0.0, 0.0, 1.5, 3.0])
score["Baixo"]       = fuzz.trimf (u_score, [2.0, 3.5, 5.0])
score["Medio"]       = fuzz.trimf (u_score, [4.0, 5.0, 6.0])
score["Alto"]        = fuzz.trimf (u_score, [5.0, 6.5, 8.0])
score["Muito Alto"]  = fuzz.trapmf(u_score, [7.0, 8.5, 10.0, 10.0])

# Base de Regras Mamdani (9 regras)

rules = [
    # Sentimento Positivo
    ctrl.Rule(sentimento["Positivo"] & rating["Alto"],   score["Muito Alto"]),
    ctrl.Rule(sentimento["Positivo"] & rating["Medio"],  score["Alto"]),
    ctrl.Rule(sentimento["Positivo"] & rating["Baixo"],  score["Medio"]),
    # Sentimento Neutro
    ctrl.Rule(sentimento["Neutro"]   & rating["Alto"],   score["Alto"]),
    ctrl.Rule(sentimento["Neutro"]   & rating["Medio"],  score["Medio"]),
    ctrl.Rule(sentimento["Neutro"]   & rating["Baixo"],  score["Baixo"]),
    # Sentimento Negativo
    ctrl.Rule(sentimento["Negativo"] & rating["Alto"],   score["Baixo"]),
    ctrl.Rule(sentimento["Negativo"] & rating["Medio"],  score["Muito Baixo"]),
    ctrl.Rule(sentimento["Negativo"] & rating["Baixo"],  score["Muito Baixo"]),
]

sistema_ctrl = ctrl.ControlSystem(rules)

# Função de inferência por linha

def inferir_score(prob_positivo: float, rating_norm: float) -> float:
    """Executa o sistema Mamdani para um par (sentimento, rating)."""
    sim = ctrl.ControlSystemSimulation(sistema_ctrl)
    sim.input["sentimento"] = float(np.clip(prob_positivo, 0.0, 1.0))
    sim.input["rating"]     = float(np.clip(rating_norm,    0.0, 1.0))
    try:
        sim.compute()
        return round(sim.output["score"], 4)
    except Exception:
        return 5.0   # valor central como fallback

# Carregar dados e calcular scores

def load_apps_catalog() -> pd.DataFrame:
    """Lê googleplaystore.csv e prepara o catálogo de apps."""
    apps_path = os.path.join(DATA_DIR, "googleplaystore.csv")
    df = pd.read_csv(apps_path, encoding="utf-8")
    df.columns = [c.strip() for c in df.columns]

    df = df.rename(columns={
        "App"      : "app",
        "Category" : "category",
        "Rating"   : "rating_raw",
        "Reviews"  : "n_reviews_store",
        "Installs" : "installs",
        "Price"    : "price_raw",
        "Genres"   : "genre",
    })

    df = df[["app", "category", "rating_raw", "n_reviews_store", "installs", "price_raw", "genre"]]

    # Limpeza de rating
    df["rating_raw"] = pd.to_numeric(df["rating_raw"], errors="coerce")
    df = df.dropna(subset=["rating_raw"])
    df = df[df["rating_raw"].between(1.0, 5.0)]

    # Normaliza rating para [0, 1]
    df["rating_norm"] = (df["rating_raw"] - 1.0) / 4.0

    # Remove duplicatas pelo nome do app (mantém primeira ocorrência)
    df = df.drop_duplicates(subset="app", keep="first").reset_index(drop=True)
    return df


def run(sentiment_df: pd.DataFrame = None) -> pd.DataFrame:
    if sentiment_df is None:
        sent_path = os.path.join(RESULTS_DIR, "sentiment_per_app.csv")
        sentiment_df = pd.read_csv(sent_path)

    catalog = load_apps_catalog()

    # Merge: só processa apps que têm sentimento calculado
    merged = catalog.merge(sentiment_df[["app", "prob_positivo", "sentiment_label"]], on="app", how="inner")
    print(f"[Camada II] {len(merged)} apps com sentimento e dados de catálogo.")

    print("[Camada II] Calculando scores fuzzy...")
    merged["score_fuzzy"] = merged.apply(
        lambda row: inferir_score(row["prob_positivo"], row["rating_norm"]), axis=1
    )

    out_path = os.path.join(RESULTS_DIR, "catalog_fuzzy.csv")
    merged.to_csv(out_path, index=False)
    print(f"[Camada II] Catálogo com scores fuzzy salvo em {out_path}")

    # Gráficos das funções de pertinência
    _plot_membership_functions()
    _plot_score_distribution(merged)

    return merged


def _plot_membership_functions():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Sentimento
    axes[0].plot(u_sentimento, sentimento["Negativo"].mf, label="Negativo", color="#C44E52")
    axes[0].plot(u_sentimento, sentimento["Neutro"].mf,   label="Neutro",   color="#CCB974")
    axes[0].plot(u_sentimento, sentimento["Positivo"].mf, label="Positivo", color="#55A868")
    axes[0].set_title("Sentimento (Entrada 1)")
    axes[0].set_xlabel("Prob. Positivo")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Rating
    axes[1].plot(u_rating, rating["Baixo"].mf, label="Baixo", color="#C44E52")
    axes[1].plot(u_rating, rating["Medio"].mf, label="Médio", color="#CCB974")
    axes[1].plot(u_rating, rating["Alto"].mf,  label="Alto",  color="#55A868")
    axes[1].set_title("Rating Normalizado (Entrada 2)")
    axes[1].set_xlabel("Rating (0–1)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # Score
    cores = ["#C44E52", "#DD8452", "#CCB974", "#55A868", "#4C72B0"]
    termos = ["Muito Baixo", "Baixo", "Medio", "Alto", "Muito Alto"]
    for t, c in zip(termos, cores):
        axes[2].plot(u_score, score[t].mf, label=t, color=c)
    axes[2].set_title("Score de Atratividade (Saída)")
    axes[2].set_xlabel("Score (0–10)")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3)

    plt.suptitle("Funções de Pertinência — Sistema Fuzzy Mamdani", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "fuzzy_membership_functions.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("[Camada II] Gráfico de funções de pertinência salvo.")


def _plot_score_distribution(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df["score_fuzzy"], bins=30, color="#4C72B0", edgecolor="white")
    ax.set_xlabel("Score Fuzzy de Atratividade")
    ax.set_ylabel("Número de Apps")
    ax.set_title("Distribuição dos Scores Fuzzy")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "fuzzy_score_distribution.png"), dpi=150)
    plt.close()
    print("[Camada II] Gráfico de distribuição de scores salvo.")


if __name__ == "__main__":
    run()