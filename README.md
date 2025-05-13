# Système RAG de Mise en Correspondance d'Emplois

Un système de Génération Augmentée par Récupération (RAG) pour faire correspondre des CV avec des offres d'emploi en utilisant la recherche sémantique et l'analyse alimentée par l'IA. Ce projet vise à surmonter les limitations des systèmes de correspondance traditionnels basés sur les mots-clés en proposant une compréhension plus fine du contenu sémantique des documents.

## Fonctionnalités

- Analyse (parsing) et analyse sémantique des CV.
- Indexation des offres d'emploi avec plusieurs stratégies d'intégration (embedding).
- Trois approches d'évaluation utilisant différents modèles d'intégration.
- Interface web pour le téléchargement de CV et la mise en correspondance d'emplois.
- Navigation et visualisation détaillées des offres d'emploi.
- Cadre d'évaluation complet.

## Comment Ça Marche

### Architecture du Système

1. **Traitement des Offres d'Emploi et des CV**

      - Ingestion de l'ensemble de données CSV (offres d'emploi et CV).
      - Résumé de texte en utilisant LLaMA3.2 via Ollama pour condenser l'information essentielle.
      - Génération d'intégrations (embeddings) avec des modèles comme BERT et Nomic Embed-Text.
      - Stockage vectoriel dans ChromaDB pour une recherche sémantique efficace.

2. **Mise en Correspondance des CV**

      - Extraction du texte des CV (formats PDF, TXT, DOCX).
      - Résumé alimenté par l'IA (LLaMA3.2).
      - Recherche de similarité sémantique entre le CV et les offres d'emploi indexées.
      - Recommandations des K meilleures offres d'emploi (Top-K).

3. **Cadre d'Évaluation**
    Le système a été évalué en utilisant plusieurs approches pour identifier la meilleure combinaison modèle d'embedding/mesure de similarité :

      - Approche 1 : BERT + Similarité Cosinus.
      - Approche 2 : Nomic Embed-Text + Similarité L2 (distance Euclidienne).
      - Approche 3 : Nomic Embed-Text + Similarité Cosinus (Approche Principale).

### Source des Données

Face à l'absence de jeux de données publics avec des correspondances explicites CV-offres, un dataset expérimental nommé `resume_job_dataset_updated.csv` a été créé manuellement. Il agrège des données de CV (inspirées d'un dataset Kaggle) et d'offres d'emploi (inspirées de publications LinkedIn). Ce dataset contient des paires CV-offre annotées manuellement comme pertinentes (`is_related = 1`) ou non (`is_related = 0`) pour permettre l'évaluation du système.

## Stack Technique

- **Intégrations (Embeddings)** : BERT-bas, Nomic Embed-Text
- **Base de Données Vectorielle (Vector DB)** : ChromaDB
- **Grand Modèle de Langage (LLM)** : Ollama (avec Llama3.2)
- **Orchestration RAG**: LangChain
- **Framework Web** : FastAPI
- **Frontend** : Bootstrap + templates Jinja2
- **Calcul de Similarité**: Scikit-learn

## Structure des Répertoires

```bash
rag_datadriven/
├── Evaluation/              # Scripts d'évaluation
│   ├── evaluation_bert.py   # Évaluation BERT + Cosinus
│   ├── evaluation_l2.py     # Évaluation Nomic + L2
│   └── evaluation_main.py   # Évaluation principale Nomic + Cosinus
├── app/                     # Application web
│   ├── static/              # Fichiers CSS/JS
│   ├── templates/           # Modèles HTML
│   ├── chroma_db/           # Stockage de la base de données vectorielle
│   └── ...                  # Code principal de l'application
├── resume_job_dataset_updated.csv  # Ensemble de données traité
└── requirements.txt         # Dépendances Python du projet
```

## Installation

### Prérequis

- Python >= 3.12
- Ollama (fonctionnant en arrière-plan avec les modèles `llama3.2` et `nomic-embed-text` téléchargés)
- Clé API Mistral (pour le traitement OCR des CV, si cette fonctionnalité est utilisée)

### Étapes d'Installation

1. **Cloner le Dépôt**

    ```bash
    git clone https://github.com/MoRoPoPi/rag_datadriven.git
    cd rag_datadriven
    ```

2. **Installer les Dépendances**
    Assurez-vous d'être à la racine du projet (`rag_datadriven`).

    ```bash
    pip install -r requirements.txt
    ```

3. **Préparer l'Ensemble de Données**

      - Placer `resume_job_dataset_updated.csv` à la racine du projet.
      - Télécharger le dataset des offres d'emploi sur Kaggle via cette url `https://www.kaggle.com/datasets/arshkon/linkedin-job-postings/data` et le placer dans `app/`

4. **Initialiser l'Index ChromaDB**
    Exécutez le script d'indexation pour traiter les offres d'emploi et créer la base de données vectorielle.

    ```bash
    python app/index.py
    ```

5. **Démarrer l'Application Web**

    ```bash
    python app/app.py
    ```

## Utilisation

### Interface Web

1. Accéder à l'application à l'adresse `http://localhost:8000` (ou le port configuré).
2. Navigation :
      - **Accueil (Home)** : Aperçu du système.
      - **Parcourir les Offres (Browse Jobs)** : Voir toutes les offres d'emploi indexées.
      - **Mise en Correspondance de CV (Resume Matching)** : Télécharger un CV pour obtenir des recommandations d'emploi.

### Téléchargement de CV

1. Visiter le point de terminaison (endpoint) `/resume`.
2. Télécharger un CV au format PDF, TXT ou DOCX.
3. Voir les emplois correspondants avec les scores de pertinence.

### Scripts d'Évaluation

Les scripts d'évaluation se trouvent dans le répertoire `Evaluation/`. Ils permettent de tester les différentes approches de matching :

1. **Approche Principale (Nomic Embed-Text + Similarité Cosinus)**

    ```bash
    python Evaluation/evaluation_main.py
    ```

2. **Approche BERT + Similarité Cosinus**

    ```bash
    python Evaluation/evaluation_bert.py
    ```

3. **Approche Nomic Embed-Text + Similarité L2**

    ```bash
    python Evaluation/evaluation_l2.py
    ```

## Configuration

### Variables d'Environnement

Certaines configurations, comme la clé API Mistral, peuvent être définies comme variables d'environnement ou directement dans le code (ex: `app/app.py`).

```python
# Exemple dans app/app.py
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "your-key-here")
OLLAMA_MODEL = "llama3.2"  # Modèle utilisé par Ollama pour le résumé
```

### Paramètres Clés

Ces paramètres sont généralement définis dans les scripts d'évaluation ou d'indexation.

```python
# Exemples de constantes (peuvent être dans les scripts d'évaluation)
CSV_PATH = "resume_job_dataset_updated.csv"  # Chemin vers le dataset
PERSIST_DIR = "./app/chroma_db"  # Répertoire de persistance pour ChromaDB
TOP_K = 20  # Nombre de recommandations à afficher/évaluer
```

(Note: Le `PERSIST_DIR` dans le README original était `./chroma_db`, mais la structure suggère que la DB est dans `app/`. Ajusté pour refléter cela, mais cela doit être cohérent avec le code réel.)

## Exigences Techniques

- **Dépendances Python**: Installer via `pip install -r requirements.txt`.
- **Modèles Ollama**: Assurez-vous qu'Ollama est installé et que les modèles suivants sont téléchargés avant l'exécution :

    ```bash
    ollama pull llama3.2
    ollama pull nomic-embed-text
    ```

- Les collections ChromaDB sont généralement créées automatiquement lors de la première exécution du script d'indexation (`app/index.py`).

## Améliorations Futures Envisagées

Plusieurs pistes d'amélioration sont envisageable :

- Enrichissement continu du jeu de données.
- Optimisation des modèles et des hyperparamètres.
- Intégration dans des plateformes de recrutement existantes.
- Ajouter un système d'authentification des utilisateurs.
- Implémenter les mises à jour d'emploi en temps réel.
- Ajouter la comparaison de plusieurs CV.
- Améliorer les fonctionnalités d'analyse des salaires.
- Ajouter un système de classement des entreprises.
