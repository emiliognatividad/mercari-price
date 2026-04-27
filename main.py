from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Mercari Price Predictor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load('model.pkl')
le_brand = joblib.load('le_brand.pkl')
le_cat1 = joblib.load('le_cat1.pkl')
le_cat2 = joblib.load('le_cat2.pkl')
cat1_values = joblib.load('cat1_values.pkl')
cat2_values = joblib.load('cat2_values.pkl')
brand_values = joblib.load('brand_values.pkl')

class Item(BaseModel):
    name: str
    item_condition_id: int  # 1-5
    brand_name: str
    cat1: str
    cat2: str
    shipping: int  # 0 or 1
    item_description: str

@app.get("/")
def root():
    return {"status": "ok"}

@app.get("/options")
def options():
    return {
        "categories": sorted(cat1_values),
        "subcategories": sorted(cat2_values),
        "brands": sorted(brand_values),
    }

@app.post("/predict")
def predict(item: Item):
    brand = item.brand_name if item.brand_name in le_brand.classes_ else 'unknown'
    cat1 = item.cat1 if item.cat1 in le_cat1.classes_ else 'unknown'
    cat2 = item.cat2 if item.cat2 in le_cat2.classes_ else 'unknown'

    brand_enc = le_brand.transform([brand])[0]
    cat1_enc = le_cat1.transform([cat1])[0]
    cat2_enc = le_cat2.transform([cat2])[0]

    name_len = len(item.name)
    desc_len = len(item.item_description)

    X = [[item.item_condition_id, item.shipping, brand_enc, cat1_enc, cat2_enc, name_len, desc_len]]
    log_pred = model.predict(X)[0]
    price = float(np.expm1(log_pred))

    return {
        "predicted_price": round(price, 2),
        "low": round(price * 0.8, 2),
        "high": round(price * 1.2, 2),
    }

@app.get("/category-map")
def category_map():
    import json
    with open('category_map.json', 'r') as f:
        return json.load(f)
