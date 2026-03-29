# Ridehail DS Platform

An end-to-end data science platform that simulates real-world analytics work at a ride-hailing company. Built to practice the full data scientist workflow — from raw data ingestion to ML-powered demand forecasting and AI-assisted analysis.

---

##  Why I Built This

Most data science projects stop at the notebook level. This project goes further — containerized with Docker, backed by a real database, and deployed as an interactive dashboard. It reflects the kind of scalable, systematic solutions that modern data teams actually build.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Containerization | Docker, Docker Compose |
| Database | PostgreSQL |
| Pipeline | Python, Pandas, SQLAlchemy |
| Dashboard | Streamlit |
| ML Model | Scikit-learn (Gradient Boosting) |
| AI Analyst | GPT-5 nano via KADA AI Cloud |

---

##  Features

- 📈 **Analytics Dashboard** — orders trend, city breakdown, promo impact, driver efficiency
- 🌦 **Segmentation** — weather impact, weekday vs weekend, day-of-week patterns
- 🤖 **Demand Forecasting** — ML model predicts order volume based on city, weather, promo, and more
- 💬 **AI Analyst** — ask business questions about the data in natural language

---

## Project Structure

```
ridehail-ds-platform/
│
├── ai_assistant/
│   └── analyst_bot.py        # CLI version of AI analyst
│
├── analytics/
│   ├── analysis.py           # Exploratory analysis scripts
│   ├── demand_model.py       # ML model training script
│   └── *.pkl                 # Saved model & encoders
│
├── dashboard/
│   └── app.py                # Streamlit dashboard
│
├── data/
│   └── orders.csv            # Raw orders dataset
│
├── pipeline/
│   ├── ingest.py             # Load CSV → PostgreSQL
│   └── transform.py          # Feature engineering → new table
│
├── .env                      # API keys (not uploaded)
├── docker-compose.yml        # Orchestrates all services
├── Dockerfile                # Container build instructions
└── requirements.txt          # Python dependencies
```

---

## How to Run

### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running
- KADA AI Cloud API key (for AI Analyst feature)

### 1. Clone the repo
```bash
git clone https://github.com/nazwaoktaviani02/ridehail-ds-platform.git
cd ridehail-ds-platform
```

### 2. Set up environment variables
Create a `.env` file in the root folder:
```env
KADA_API_KEY=your_api_key_here
KADA_BASE_URL=your_base_url_here
```

### 3. Run with Docker
```bash
docker compose up --build
```

### 4. Open the dashboard
Go to [http://localhost:8501](http://localhost:8501)

---

## How It Works

```
orders.csv
    ↓ ingest.py         → loads raw data into PostgreSQL
    ↓ transform.py      → adds derived metrics (e.g. orders per driver)
    ↓ demand_model.py   → trains ML model, saves as .pkl
    ↓ app.py            → Streamlit dashboard reads from DB + loads model
```

All services run in Docker containers and communicate through Docker Compose's internal network — no manual setup needed.

---

## ML Model

- **Algorithm:** Gradient Boosting Regressor
- **Target:** Predicted number of orders
- **Features:** city, weather, promo, drivers online, day of week, month, is weekend
- **Evaluation:** MAE, RMSE, R²

---
