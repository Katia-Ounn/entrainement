# Time Series ML Training Platform — Cevital

Plateforme expérimentale de **Maintenance Prédictive** basée sur des séries temporelles.  
Entraînement de modèles LSTM / GRU pour la prédiction du **RUL** (Remaining Useful Life) à partir des données GMAO Cevital.

---

## Prérequis

- **Python** 3.10 ou 3.11 → [python.org](https://www.python.org/downloads/)
- **Node.js** 18+ → [nodejs.org](https://nodejs.org/)
- **Git** → [git-scm.com](https://git-scm.com/)

---

## Installation

### 1. Cloner le projet

```bash
git clone https://github.com/Katia-Ounn/entrainement.git
cd entrainement
```

---

### 2. Backend (FastAPI)

```bash
cd backend

# Créer l'environnement virtuel
python -m venv venv

# Activer l'environnement (Windows)
venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt
```

#### Lancer le backend

```bash
# Toujours dans le dossier backend, venv activé
uvicorn main:app --reload --port 8000
```

Le backend tourne sur → **http://localhost:8000**

---

### 3. Frontend (React + Vite)

Ouvrir un **nouveau terminal** :

```bash
cd frontend

# Installer les dépendances
npm install

# Lancer le frontend
npm run dev
```

Le frontend tourne sur → **http://localhost:5173**

---

## Utilisation

1. Ouvrir **http://localhost:5173** dans le navigateur
2. Aller dans **Data Preparation** → uploader `failure.csv` + `equipment.csv`
3. Lancer l'EDA, le Feature Engineering, puis le Preprocessing
4. Aller dans **Training** → choisir Manuel ou AutoML (Bayésien)
5. Voir les résultats dans **Leaderboard**

---

## Structure du projet

```
entrainement/
├── backend/
│   ├── main.py                  # API FastAPI (routes + WebSocket)
│   ├── tuner.py                 # Entraînement LSTM/GRU + Keras Tuner
│   ├── models.py                # Modèles SQLAlchemy (Dataset, Experiment)
│   ├── requirements.txt         # Dépendances Python
│   ├── pipelines/
│   │   └── cevital_pipeline.py  # Pipeline complet (EDA, Features, Preprocessing)
│   └── demos/                   # Démos pédagogiques
└── frontend/
    ├── src/
    │   ├── App.jsx              # Application principale
    │   ├── AppContext.jsx       # État global
    │   └── components/
    │       ├── prep/            # Data Preparation (EDA, Features, Preprocessing)
    │       ├── TrainingPanel.jsx
    │       └── Leaderboard.jsx
    └── package.json
```

---

## Technologies

| Couche | Technologies |
|--------|-------------|
| Backend | FastAPI · SQLAlchemy · SQLite · WebSocket |
| ML | TensorFlow · Keras · Keras Tuner (Bayesian) |
| Frontend | React · Vite · Tailwind CSS · Recharts |
| Modèle | LSTM / GRU · Embedding composant · Régression RUL |
