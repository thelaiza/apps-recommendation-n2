"""
main.py — Pipeline Principal do Sistema Inteligente de Recomendação de Apps
---------------------------------------------------------------------------
Executa as três camadas em sequência, passando a saída de cada uma como
entrada da próxima:

  Camada I  → PLN + Naive Bayes     probabilidade de sentimento por app
  Camada II → Fuzzy Mamdani         score de atratividade por app
  Camada III→ Algoritmo Genético    portfólio de 5 apps recomendados

Uso:
  python main.py               # executa o pipeline completo
  python main.py --skip-layer1 # pula a Camada I (usa results/ existentes)
  python main.py --skip-layer2 # pula até a Camada II
"""

import os
import sys
import time

# Ajusta path para importar os módulos de src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import layer1_nlp   as camada1
import layer2_fuzzy as camada2
import layer3_ga    as camada3

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def banner(texto: str):
    line = "─" * 60
    print(f"\n{line}")
    print(f"  {texto}")
    print(f"{line}")


def check_data():
    """Verifica se os arquivos do dataset estão presentes."""
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    required = [
        "googleplaystore.csv",
        "googleplaystore_user_reviews.csv",
    ]
    missing = [f for f in required if not os.path.exists(os.path.join(data_dir, f))]
    if missing:
        print("\n[ERRO] Arquivos ausentes na pasta data/:")
        for f in missing:
            print(f"  - {f}")
        print(
            "\nBaixe o dataset em:\n"
            "  https://www.kaggle.com/datasets/lava18/google-play-store-apps\n"
            "e extraia os CSVs para a pasta data/.\n"
        )
        sys.exit(1)


def main():
    skip_layer1 = "--skip-layer1" in sys.argv or "--skip-layer2" in sys.argv
    skip_layer2 = "--skip-layer2" in sys.argv

    check_data()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    t0 = time.time()

    # Camada I 
    if skip_layer1:
        banner("Camada I — PLN + Naive Bayes [PULADA]")
        import pandas as pd
        sentiment_df = pd.read_csv(os.path.join(RESULTS_DIR, "sentiment_per_app.csv"))
        print(f"  Lidos {len(sentiment_df)} apps de results/sentiment_per_app.csv")
    else:
        banner("Camada I — PLN + Naive Bayes")
        sentiment_df = camada1.run()

    # Camada II
    if skip_layer2:
        banner("Camada II — Sistema Fuzzy Mamdani [PULADA]")
        import pandas as pd
        catalog_df = pd.read_csv(os.path.join(RESULTS_DIR, "catalog_fuzzy.csv"))
        print(f"  Lidos {len(catalog_df)} apps de results/catalog_fuzzy.csv")
    else:
        banner("Camada II — Sistema Fuzzy Mamdani")
        catalog_df = camada2.run(sentiment_df)

    # Camada III
    banner("Camada III — Algoritmo Genético")
    recommended = camada3.run(catalog_df)

    # Resumo final
    elapsed = time.time() - t0
    banner(f"Pipeline concluído em {elapsed:.1f}s")

    print("\nArquivos gerados em results/:")
    for fname in sorted(os.listdir(RESULTS_DIR)):
        fpath = os.path.join(RESULTS_DIR, fname)
        size  = os.path.getsize(fpath)
        print(f"  {fname:<40} {size/1024:>7.1f} KB")

    print("\n✔ Recomendação final:")
    if "app" in recommended.columns and "category" in recommended.columns:
        for _, row in recommended.iterrows():
            print(f"  • {row['app']:<45} [{row['category']}]  score={row.get('score_fuzzy', '?'):.2f}")


if __name__ == "__main__":
    main()