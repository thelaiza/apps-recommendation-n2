# N2 — Sistema Inteligente de Recomendação de Apps

**Disciplina:** Inteligência Artificial  
**Professor:** Claudinei Dias (Ney)  
**Instituição:** Centro Universitário Católica de Santa Catarina

---

## Integrantes

- Jhessica Alves
- Laíza Silva

---

## Objetivo

O projeto utiliza o dataset público do Google Play Store para recomendar aplicativos com base nas avaliações dos usuários. As três técnicas de IA são encadeadas de forma que a saída de cada camada alimenta a próxima:

1. **PLN + Naive Bayes** — lê os textos das reviews, realiza pré-processamento e classifica o sentimento (Positivo, Negativo ou Neutro)
2. **Sistema de Inferência Fuzzy Mamdani** — combina o sentimento com o rating médio do app para gerar um score de atratividade
3. **Algoritmo Genético** — seleciona a melhor combinação de 5 apps, maximizando score e diversidade de categorias

---

## Dataset

**Google Play Store Apps** — dataset público disponível no Kaggle com dados de mais de 10.000 apps e suas respectivas avaliações.

🔗 https://www.kaggle.com/datasets/lava18/google-play-store-apps

Arquivos utilizados (colocar na pasta `data/`):

| Arquivo | Conteúdo |
|---------|----------|
| `googleplaystore.csv` | Catálogo de apps com rating, categoria, preço etc. |
| `googleplaystore_user_reviews.csv` | Reviews dos usuários com polarity anotada |

---

## Arquitetura do Sistema

```
Reviews dos usuários (texto livre)
         │
         ▼
┌──────────────────────────────┐
│   CAMADA I — PLN             │  Tokenização · Stop Words EN · Stemming
│   Naive Bayes (TF-IDF)       │  Classificação: Positivo / Negativo / Neutro
└──────────┬───────────────────┘
           │ prob_positivo por app
           ▼
┌──────────────────────────────┐
│   CAMADA II — FUZZY          │  Entradas: sentimento + rating normalizado
│   Mamdani (scikit-fuzzy)     │  Saída: Score de Atratividade [0–10]
└──────────┬───────────────────┘
           │ score_fuzzy por app
           ▼
┌──────────────────────────────┐
│   CAMADA III — GA            │  Seleciona combo de 5 apps
│   Algoritmo Genético (DEAP)  │  Maximiza score · diversidade · sem repetição
└──────────┬───────────────────┘
           │
           ▼
    Recomendação final (5 apps)
```

---

## Camada I — PLN + Naive Bayes

### Pré-processamento

1. **Tokenização** — `word_tokenize` do NLTK
2. **Remoção de stop words** — lista inglesa do NLTK
3. **Stemming** — algoritmo Snowball para inglês
4. **Vetorização** — TF-IDF com até 8.000 features e bigramas (1, 2)

### Rotulagem

O dataset já inclui o campo `Sentiment` (Positive / Negative / Neutral), que é mapeado para Positivo / Negativo / Neutro.

### Saída

Probabilidade média de sentimento positivo por app → `results/sentiment_per_app.csv`

---

## Camada II — Sistema de Inferência Fuzzy

### Variáveis e Funções de Pertinência

**Entrada 1 — Sentimento** (prob. positivo, [0, 1])

| Termo | Tipo | Parâmetros |
|-------|------|-----------|
| Negativo | Trapezoidal | [0, 0, 0.25, 0.45] |
| Neutro | Triangular | [0.35, 0.50, 0.65] |
| Positivo | Trapezoidal | [0.55, 0.75, 1.0, 1.0] |

**Entrada 2 — Rating Normalizado** ((rating−1)/4, [0, 1])

| Termo | Tipo | Parâmetros |
|-------|------|-----------|
| Baixo | Trapezoidal | [0, 0, 0.25, 0.45] |
| Médio | Triangular | [0.35, 0.50, 0.65] |
| Alto | Trapezoidal | [0.55, 0.75, 1.0, 1.0] |

**Saída — Score de Atratividade** ([0, 10])

| Termo | Tipo | Parâmetros |
|-------|------|-----------|
| Muito Baixo | Trapezoidal | [0, 0, 1.5, 3.0] |
| Baixo | Triangular | [2.0, 3.5, 5.0] |
| Médio | Triangular | [4.0, 5.0, 6.0] |
| Alto | Triangular | [5.0, 6.5, 8.0] |
| Muito Alto | Trapezoidal | [7.0, 8.5, 10, 10] |

### Base de Regras (9 regras Mamdani)

| SE sentimento... | E rating... | ENTÃO score... |
|-----------------|-------------|---------------|
| Positivo | Alto | Muito Alto |
| Positivo | Médio | Alto |
| Positivo | Baixo | Médio |
| Neutro | Alto | Alto |
| Neutro | Médio | Médio |
| Neutro | Baixo | Baixo |
| Negativo | Alto | Baixo |
| Negativo | Médio | Muito Baixo |
| Negativo | Baixo | Muito Baixo |

---

## Camada III — Algoritmo Genético

### Configuração

| Parâmetro | Valor |
|-----------|-------|
| Tamanho da população | 120 indivíduos |
| Número de gerações | 100 |
| Probabilidade de crossover | 80% |
| Probabilidade de mutação | 20% |
| Seleção | Torneio (k=4) |
| Crossover | Ordenado (OX1) |
| Tamanho do portfólio | 5 apps |
| Máximo por categoria | 2 apps |

### Representação

Cada **indivíduo** é um vetor de 5 índices inteiros únicos do catálogo de apps candidatos (sem repetição de índice).

### Função de Fitness

```
fitness = Σ(score_fuzzy) + bônus_diversidade − penalidade_repetição

bônus_diversidade = 2.0 × (n_categorias_distintas − 1)
penalidade_repetição = 3.0 × max(0, apps_mesma_categoria − 2)
```

---

## Como Executar

### 1. Pré-requisitos

- Python 3.10+
- Dataset na pasta `data/` (ver link acima)

### 2. Instalar dependências

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
# .venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

### 3. Executar o pipeline completo

```bash
python main.py
```

### 4. Executar camadas individualmente

```bash
# Apenas Camada I
python src/layer1_nlp.py

# Apenas Camada II (requer saída da Camada I)
python src/layer2_fuzzy.py

# Apenas Camada III (requer saída da Camada II)
python src/layer3_ga.py
```

### 5. Pular camadas já executadas

```bash
python main.py --skip-layer1   # pula Camada I e II, reusa CSVs
python main.py --skip-layer2   # pula só Camada I
```

### 6. Arquivos gerados em `results/`

| Arquivo | Descrição |
|---------|-----------|
| `modelo_nb.pkl` | Modelo Naive Bayes treinado |
| `sentiment_per_app.csv` | Probabilidade de sentimento por app |
| `catalog_fuzzy.csv` | Catálogo com score fuzzy de cada app |
| `recommended_apps.csv` | 5 apps recomendados pelo GA |
| `nb_confusion_matrix.png` | Matriz de confusão do Naive Bayes |
| `sentiment_distribution.png` | Distribuição de sentimentos |
| `fuzzy_membership_functions.png` | Funções de pertinência |
| `fuzzy_score_distribution.png` | Distribuição dos scores fuzzy |
| `ga_evolution.png` | Curva de evolução do fitness por geração |

---

## Estrutura do Projeto

```
apps-recommendation-n2/
├── data/                              # Dataset (não versionado)
│   ├── googleplaystore.csv
│   └── googleplaystore_user_reviews.csv
├── src/
│   ├── layer1_nlp.py                  # Camada I: PLN + Naive Bayes
│   ├── layer2_fuzzy.py                # Camada II: Fuzzy Mamdani
│   └── layer3_ga.py                   # Camada III: Algoritmo Genético
├── results/                           # Saídas geradas (não versionado)
├── main.py                            # Pipeline completo
├── requirements.txt
└── README.md
```

---

## Justificativa das Escolhas Tecnológicas

**Camada II — Mamdani vs. Sugeno:** O método Mamdani produz conjuntos fuzzy como saída antes da defuzzificação, tornando o processo de inferência mais interpretável e facilitando a visualização das funções de pertinência — aspecto relevante tanto para a validação do sistema quanto para a apresentação dos resultados. O modelo Sugeno, apesar de computacionalmente mais eficiente, produz saídas como funções lineares, reduzindo a transparência das regras para fins didáticos.

**Camada III — GA vs. ACO/PSO/SA:** O problema de seleção de apps para recomendação é naturalmente representado como um problema de otimização combinatória com variáveis discretas — cada app pode ou não integrar o portfólio final. Essa estrutura se alinha diretamente com a representação cromossômica do GA (vetor de índices inteiros únicos). O crossover ordenado (OX1) e a mutação por troca exploram eficientemente o espaço de soluções respeitando a restrição de unicidade de apps no portfólio.

---

## Tecnologias Utilizadas

| Biblioteca | Finalidade |
|------------|-----------|
| `scikit-learn` | Naive Bayes, TF-IDF, métricas de avaliação |
| `nltk` | Tokenização, stop words, stemming Snowball |
| `scikit-fuzzy` | Sistema de inferência fuzzy Mamdani |
| `deap` | Framework para algoritmos evolutivos |
| `pandas` | Manipulação e análise dos dados |
| `numpy` | Operações numéricas |
| `matplotlib` | Visualizações e gráficos |

---

## Referências

- Google Play Store Apps Dataset. Kaggle. Disponível em: https://www.kaggle.com/datasets/lava18/google-play-store-apps
- ZADEH, L. A. Fuzzy sets. *Information and Control*, v. 8, n. 3, p. 338–353, 1965.
- GOLDBERG, D. E. *Genetic Algorithms in Search, Optimization and Machine Learning*. Addison-Wesley, 1989.
- MITCHELL, T. M. *Machine Learning*. McGraw-Hill, 1997.
