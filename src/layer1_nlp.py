"""
Camada I — Percepção e Sentimento (PLN + Naive Bayes)
------------------------------------------------------
Lê as reviews do Google Play Store, aplica pré-processamento textual e
treina um classificador Naive Bayes para análise de sentimentos.

Saída: results/sentiment_per_app.csv
  - app           : nome do aplicativo
  - prob_positivo : probabilidade média de sentimento positivo
  - sentiment_label: rótulo majoritário (Positivo / Neutro / Negativo)
"""

import os
import re
import pickle
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# Caminhos
DATA_DIR    = os.path.join(os.path.dirname(__file__), "..", "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

REVIEWS_FILE = os.path.join(DATA_DIR, "googleplaystore_user_reviews.csv")
APPS_FILE    = os.path.join(DATA_DIR, "googleplaystore.csv")

# Download dos recursos NLTK
for resource in ["punkt", "stopwords", "punkt_tab"]:
    try:
        nltk.data.find(f"tokenizers/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)

STOP_EN  = set(stopwords.words("english"))
STEMMER  = SnowballStemmer("english")

# Pré-processamento textual

def preprocess(text: str) -> str:
    """Tokenização → remoção de stop words → stemming."""
    if not isinstance(text, str) or text.strip() == "":
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)          # remove pontuação / números
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in STOP_EN and len(t) > 2]
    tokens = [STEMMER.stem(t) for t in tokens]
    return " ".join(tokens)

# Rotulagem de sentimento

def label_sentiment(polarity: str) -> str:
    """Mapeia Sentiment do dataset para Positivo / Neutro / Negativo."""
    mapping = {
        "Positive": "Positivo",
        "Negative": "Negativo",
        "Neutral" : "Neutro",
    }
    return mapping.get(polarity, None)

# Carregamento e limpeza

def load_data() -> pd.DataFrame:
    print("[Camada I] Carregando reviews...")
    df = pd.read_csv(REVIEWS_FILE, encoding="utf-8")

    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={
        "App"              : "app",
        "Translated_Review": "review",
        "Sentiment"        : "sentiment_raw",
    })

    df = df[["app", "review", "sentiment_raw"]].dropna()
    df["sentiment"] = df["sentiment_raw"].apply(label_sentiment)
    df = df.dropna(subset=["sentiment"])

    df["review_clean"] = df["review"].apply(preprocess)
    df = df[df["review_clean"].str.strip() != ""]

    print(f"[Camada I] {len(df):,} reviews válidas carregadas.")
    print(df["sentiment"].value_counts())
    return df

# Treinamento do modelo

def train_model(df: pd.DataFrame):
    X = df["review_clean"]
    y = df["sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=8000,
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
        )),
        ("nb", MultinomialNB(alpha=0.5)),
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    print("\n[Camada I] Relatório de Classificação:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Matriz de confusão
    cm = confusion_matrix(y_test, y_pred, labels=["Positivo", "Neutro", "Negativo"])
    disp = ConfusionMatrixDisplay(cm, display_labels=["Positivo", "Neutro", "Negativo"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title("Matriz de Confusão — Naive Bayes")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "nb_confusion_matrix.png"), dpi=150)
    plt.close()
    print("[Camada I] Matriz de confusão salva.")

    # Salva modelo
    model_path = os.path.join(RESULTS_DIR, "modelo_nb.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"[Camada I] Modelo salvo em {model_path}")

    return pipeline

# Cálculo de probabilidade por app

def compute_app_sentiment(df: pd.DataFrame, pipeline) -> pd.DataFrame:
    """Calcula a probabilidade média de sentimento positivo por app."""
    classes = list(pipeline.classes_)
    idx_pos = classes.index("Positivo")

    probs = pipeline.predict_proba(df["review_clean"])
    df = df.copy()
    df["prob_positivo"] = probs[:, idx_pos]

    agg = df.groupby("app").agg(
        prob_positivo=("prob_positivo", "mean"),
        n_reviews=("review_clean", "count"),
    ).reset_index()

    # Remove apps com poucas reviews (ruído)
    agg = agg[agg["n_reviews"] >= 3].copy()

    # Rótulo majoritário por app
    df["pred"] = pipeline.predict(df["review_clean"])
    label_df = (
        df.groupby(["app", "pred"])
        .size()
        .reset_index(name="cnt")
        .sort_values("cnt", ascending=False)
        .drop_duplicates("app")
        .rename(columns={"pred": "sentiment_label"})
    )
    agg = agg.merge(label_df[["app", "sentiment_label"]], on="app", how="left")

    out_path = os.path.join(RESULTS_DIR, "sentiment_per_app.csv")
    agg.to_csv(out_path, index=False)
    print(f"[Camada I] Sentimento por app salvo em {out_path} ({len(agg)} apps)")
    return agg

# Plot distribuição de sentimentos

def plot_sentiment_distribution(agg: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Histograma de prob_positivo
    axes[0].hist(agg["prob_positivo"], bins=30, color="#4C72B0", edgecolor="white")
    axes[0].set_xlabel("Probabilidade de Sentimento Positivo")
    axes[0].set_ylabel("Número de Apps")
    axes[0].set_title("Distribuição da Prob. Positiva por App")
    axes[0].grid(axis="y", alpha=0.3)

    # Pizza de rótulos
    counts = agg["sentiment_label"].value_counts()
    colors = {"Positivo": "#55A868", "Neutro": "#CCB974", "Negativo": "#C44E52"}
    axes[1].pie(
        counts.values,
        labels=counts.index,
        colors=[colors.get(l, "#888") for l in counts.index],
        autopct="%1.1f%%",
        startangle=90,
    )
    axes[1].set_title("Rótulo de Sentimento por App")

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "sentiment_distribution.png"), dpi=150)
    plt.close()
    print("[Camada I] Gráfico de distribuição salvo.")

# Ponto de entrada

def run():
    df       = load_data()
    pipeline = train_model(df)
    agg      = compute_app_sentiment(df, pipeline)
    plot_sentiment_distribution(agg)
    return agg


if __name__ == "__main__":
    run()