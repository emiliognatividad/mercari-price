import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
import joblib

print("Loading data...")
df = pd.read_csv('train.tsv', sep='\t')

# Clean
df = df[df['price'] > 0].copy()
df['brand_name'] = df['brand_name'].fillna('unknown')
df['category_name'] = df['category_name'].fillna('unknown')
df['item_description'] = df['item_description'].fillna('')

# Split category into levels
df['cat1'] = df['category_name'].apply(lambda x: x.split('/')[0] if '/' in x else x)
df['cat2'] = df['category_name'].apply(lambda x: x.split('/')[1] if x.count('/') >= 1 else 'unknown')

# Encode categoricals
le_brand = LabelEncoder()
le_cat1 = LabelEncoder()
le_cat2 = LabelEncoder()

df['brand_enc'] = le_brand.fit_transform(df['brand_name'])
df['cat1_enc'] = le_cat1.fit_transform(df['cat1'])
df['cat2_enc'] = le_cat2.fit_transform(df['cat2'])

# Features
features = ['item_condition_id', 'shipping', 'brand_enc', 'cat1_enc', 'cat2_enc', 'name_len', 'desc_len']
df['name_len'] = df['name'].apply(len)
df['desc_len'] = df['item_description'].apply(len)

X = df[features]
y = np.log1p(df['price'])  # log transform price

print("Training model...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.1, n_jobs=-1)
model.fit(X_train, y_train)

preds = np.expm1(model.predict(X_test))
actual = np.expm1(y_test)
mae = mean_absolute_error(actual, preds)
print(f"MAE: ${mae:.2f}")

# Save
joblib.dump(model, 'model.pkl')
joblib.dump(le_brand, 'le_brand.pkl')
joblib.dump(le_cat1, 'le_cat1.pkl')
joblib.dump(le_cat2, 'le_cat2.pkl')
joblib.dump(df['cat1'].unique().tolist(), 'cat1_values.pkl')
joblib.dump(df['cat2'].unique().tolist(), 'cat2_values.pkl')
joblib.dump(df['brand_name'].unique().tolist(), 'brand_values.pkl')
print("Model saved.")
