"""
Camada III — Otimização Estocástica (Algoritmo Genético)
---------------------------------------------------------
Seleciona a melhor combinação de N apps para recomendar a um usuário,
maximizando o score fuzzy total e a diversidade de categorias,
respeitando a restrição de que nenhuma categoria se repita mais de
MAX_PER_CATEGORY vezes no portfólio.

Representação cromossômica:
  Vetor de inteiros de comprimento PORTFOLIO_SIZE, onde cada gene é um
  índice único no catálogo de apps com score fuzzy calculado.

Função de Fitness:
  fitness = Σ(score_fuzzy) + bônus_diversidade − penalidade_repetição

Resultado: results/recommended_apps.csv
           results/ga_evolution.png
"""

import os
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms

warnings.filterwarnings("ignore")

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Hiperparâmetros do GA

PORTFOLIO_SIZE    = 5      # quantidade de apps recomendados
POP_SIZE          = 120    # tamanho da população
N_GENERATIONS     = 100    # número de gerações
CX_PROB           = 0.80   # probabilidade de crossover
MUT_PROB          = 0.20   # probabilidade de mutação
TOURNAMENT_K      = 4      # tamanho do torneio de seleção
MAX_PER_CATEGORY  = 2      # máximo de apps da mesma categoria por portfólio
DIVERSITY_BONUS   = 2.0    # bônus por categoria distinta adicional
REPEAT_PENALTY    = 3.0    # penalidade por cada app repetido além do limite

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Catálogo global (preenchido em run())

_catalog: pd.DataFrame = None

# Fitness

def fitness(individual):
    """Avalia um indivíduo (lista de índices) e retorna (fitness,)."""
    global _catalog

    # Penalidade por genes duplicados (cromossomo inválido)
    if len(set(individual)) < len(individual):
        return (-999.0,)

    rows = _catalog.iloc[individual]

    score_total = rows["score_fuzzy"].sum()

    # Bônus de diversidade: cada categoria distinta além da primeira
    n_cats = rows["category"].nunique()
    diversidade = DIVERSITY_BONUS * (n_cats - 1)

    # Penalidade por categoria repetida acima do limite
    cat_counts = rows["category"].value_counts()
    over_limit = (cat_counts - MAX_PER_CATEGORY).clip(lower=0).sum()
    penalidade = REPEAT_PENALTY * over_limit

    return (score_total + diversidade - penalidade,)

# Operadores genéticos customizados

def cx_ordered(ind1, ind2):
    """Crossover ordenado (OX1) que preserva genes únicos."""
    size = len(ind1)
    a, b = sorted(random.sample(range(size), 2))

    # Segmento do pai 1
    child1_mid = ind1[a:b]
    child2_mid = ind2[a:b]

    # Complemento do pai 2 (sem os que já estão no segmento)
    fill1 = [g for g in ind2 if g not in child1_mid]
    fill2 = [g for g in ind1 if g not in child2_mid]

    ind1[:] = fill1[:a] + child1_mid + fill1[a:]
    ind2[:] = fill2[:a] + child2_mid + fill2[a:]

    # Garante tamanho correto
    ind1[:] = ind1[:size]
    ind2[:] = ind2[:size]
    return ind1, ind2


def mut_swap(individual, catalog_size: int):
    """Mutação: troca um gene por um índice aleatório não presente no indivíduo."""
    pos = random.randint(0, len(individual) - 1)
    available = list(set(range(catalog_size)) - set(individual))
    if available:
        individual[pos] = random.choice(available)
    return (individual,)

# Setup do DEAP

def _setup_deap(catalog_size: int):
    # Limpa registros anteriores (evita erros em re-execuções)
    if "FitnessMax" in creator.__dict__:
        del creator.FitnessMax
    if "Individual" in creator.__dict__:
        del creator.Individual

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    def rand_individual():
        return creator.Individual(random.sample(range(catalog_size), PORTFOLIO_SIZE))

    toolbox.register("individual",   tools.initIterate, creator.Individual, rand_individual)
    toolbox.register("population",   tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate",     fitness)
    toolbox.register("select",       tools.selTournament, tournsize=TOURNAMENT_K)
    toolbox.register("mate",         cx_ordered)
    toolbox.register("mutate",       mut_swap, catalog_size=catalog_size)

    return toolbox

# Execução do GA

def run_ga(catalog: pd.DataFrame):
    global _catalog
    _catalog = catalog.reset_index(drop=True)
    catalog_size = len(_catalog)

    toolbox = _setup_deap(catalog_size)

    population = toolbox.population(n=POP_SIZE)
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("max",  np.max)
    stats.register("mean", np.mean)
    stats.register("min",  np.min)

    hof = tools.HallOfFame(1)

    best_per_gen = []
    avg_per_gen  = []

    print(f"[Camada III] Iniciando GA: {POP_SIZE} indivíduos, {N_GENERATIONS} gerações.")

    for gen in range(N_GENERATIONS):
        # Seleção
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        # Crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CX_PROB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # Mutação
        for mutant in offspring:
            if random.random() < MUT_PROB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Avaliação dos inválidos
        invalid = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid:
            ind.fitness.values = toolbox.evaluate(ind)

        population[:] = offspring
        hof.update(population)

        record = stats.compile(population)
        best_per_gen.append(record["max"])
        avg_per_gen.append(record["mean"])

        if (gen + 1) % 20 == 0:
            print(f"  Geração {gen+1:3d}/{N_GENERATIONS} | "
                  f"Melhor={record['max']:.4f} | Média={record['mean']:.4f}")

    best_individual = hof[0]
    print(f"\n[Camada III] Melhor fitness: {best_individual.fitness.values[0]:.4f}")

    return best_individual, best_per_gen, avg_per_gen

# Resultado final

def build_recommendation(best_individual, catalog: pd.DataFrame) -> pd.DataFrame:
    rows = catalog.iloc[best_individual].copy()
    cols = ["app", "category", "rating_raw", "prob_positivo", "score_fuzzy", "sentiment_label"]
    cols = [c for c in cols if c in rows.columns]
    rec = rows[cols].reset_index(drop=True)

    out_path = os.path.join(RESULTS_DIR, "recommended_apps.csv")
    rec.to_csv(out_path, index=False)
    print(f"[Camada III] Recomendação final salva em {out_path}")

    print("\n===== RECOMENDAÇÃO FINAL =====")
    print(rec.to_string(index=False))
    print(f"\nScore total: {rec['score_fuzzy'].sum():.4f}")
    print(f"Categorias distintas: {rec['category'].nunique()}")
    return rec

# Gráfico de evolução

def plot_evolution(best_per_gen, avg_per_gen):
    fig, ax = plt.subplots(figsize=(9, 4))
    gens = range(1, len(best_per_gen) + 1)
    ax.plot(gens, best_per_gen, label="Melhor fitness", color="#55A868", linewidth=2)
    ax.plot(gens, avg_per_gen,  label="Fitness médio",  color="#4C72B0", linewidth=1.5, linestyle="--")
    ax.set_xlabel("Geração")
    ax.set_ylabel("Fitness")
    ax.set_title("Evolução do Algoritmo Genético")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "ga_evolution.png"), dpi=150)
    plt.close()
    print("[Camada III] Gráfico de evolução salvo.")

# Ponto de entrada

def run(catalog: pd.DataFrame = None) -> pd.DataFrame:
    if catalog is None:
        fuzzy_path = os.path.join(RESULTS_DIR, "catalog_fuzzy.csv")
        catalog = pd.read_csv(fuzzy_path)

    # Filtra apps com score >= 4 para reduzir o espaço de busca a candidatos relevantes
    candidates = catalog[catalog["score_fuzzy"] >= 4.0].reset_index(drop=True)
    if len(candidates) < PORTFOLIO_SIZE:
        candidates = catalog.nlargest(max(PORTFOLIO_SIZE * 4, 50), "score_fuzzy").reset_index(drop=True)

    print(f"[Camada III] {len(candidates)} apps candidatos para o GA.")

    best_ind, best_per_gen, avg_per_gen = run_ga(candidates)
    rec = build_recommendation(best_ind, candidates)
    plot_evolution(best_per_gen, avg_per_gen)
    return rec


if __name__ == "__main__":
    run()