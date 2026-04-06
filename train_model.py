import sqlite3
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle
import os

print("1. Загружаем данные из metrics...")
conn = sqlite3.connect('data/traffic.db')
df = pd.read_sql_query("SELECT minute_bucket, total_vehicles FROM metrics ORDER BY minute_bucket", conn)
conn.close()

if df.empty:
    print("❌ Нет данных! Сначала запусти run_full_pipeline.py")
    exit()

print(f"   Загружено {len(df)} записей")

# Преобразуем время
df['minute_bucket'] = pd.to_datetime(df['minute_bucket'])
min_time = df['minute_bucket'].min()
df['minute_num'] = (df['minute_bucket'] - min_time).dt.total_seconds() / 60
df['hour'] = df['minute_bucket'].dt.hour
df['dayofweek'] = df['minute_bucket'].dt.dayofweek

# Признаки и цель
X = df[['minute_num', 'hour', 'dayofweek']]
y = df['total_vehicles']

print("2. Обучаем модель линейной регрессии...")
model = LinearRegression()
model.fit(X, y)

# Оценка
y_pred = model.predict(X)
mae = mean_absolute_error(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))
print(f"   ✅ Модель обучена. MAE = {mae:.2f}, RMSE = {rmse:.2f}")

# Сохраняем модель и метаданные
os.makedirs('models', exist_ok=True)
with open('models/traffic_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('models/metadata.pkl', 'wb') as f:
    pickle.dump({'min_time': min_time}, f)

print("3. Модель сохранена в папке models/")