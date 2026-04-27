# Mercari Price Predictor

ML-powered price suggestion tool trained on 1.4 million Mercari product listings. Input item details, get a predicted selling price with confidence range and market distribution chart.

## Live Demo

- Frontend: http://54.172.217.112:3001
- API: http://54.172.217.112:8001
- API Docs: http://54.172.217.112:8001/docs

## Features

- Price prediction based on item name, brand, category, condition, and description
- Confidence range (low/high)
- Price distribution chart of similar listings with hover tooltips
- Searchable dropdowns for brand and category (5000+ options)
- Category-aware subcategory filtering

## Model

- Algorithm: XGBoost regression
- Training data: 1,482,535 listings
- Features: item condition, shipping, brand, category, subcategory, name length, description length
- MAE: $13.03 on test set (avg listing price $27)

## Stack

Python, FastAPI, XGBoost, scikit-learn, pandas, React, AWS EC2, nginx

## Running locally

**Backend**
```bash
cd mercari-price
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```

Requires `train.tsv` from the Mercari Price Suggestion Kaggle dataset to retrain. Pre-trained model files are not included in the repo.

**Frontend**
```bash
cd mercari-frontend
npm install
REACT_APP_API_URL=http://localhost:8001 npm start
```

## API endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /predict | Get price prediction |
| GET | /options | All categories, subcategories, brands |
| GET | /category-map | Category to subcategory mapping |
| GET | /price-distribution | Price histogram for a category |
