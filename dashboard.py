import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Транспортный мониторинг", layout="wide")
st.title("🚗 Интеллектуальная платформа мониторинга транспортных потоков")

# Загрузка модели и метаданных
@st.cache_resource
def load_model():
    with open('models/traffic_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    return model, metadata

model, metadata = load_model()
min_time = metadata['min_time']

# Подключение к БД
@st.cache_data(ttl=60)
def load_data():
    conn = sqlite3.connect('data/traffic.db')
    df = pd.read_sql_query("SELECT minute_bucket, total_vehicles, avg_confidence FROM metrics ORDER BY minute_bucket", conn)
    conn.close()
    df['minute_bucket'] = pd.to_datetime(df['minute_bucket'])
    return df

df = load_data()

if df.empty:
    st.error("Нет данных. Запустите детекцию и расчёт метрик.")
    st.stop()

# 1. Текущее состояние (последние 5 минут)
last_time = df['minute_bucket'].max()
last_5min = df[df['minute_bucket'] >= last_time - timedelta(minutes=5)]
current_intensity = last_5min['total_vehicles'].mean() if not last_5min.empty else 0

col1, col2, col3 = st.columns(3)
col1.metric("Текущая интенсивность (ср. за 5 мин)", f"{current_intensity:.1f} авт/мин")
col2.metric("Последнее обновление", last_time.strftime("%H:%M:%S"))
col3.metric("Всего детекций", df['total_vehicles'].sum())

# 2. График исторической интенсивности
st.subheader("📈 История интенсивности")
fig = px.line(df, x='minute_bucket', y='total_vehicles', title="Количество ТС по минутам")
st.plotly_chart(fig, use_container_width=True)

# 3. Прогнозирование на 30/60/120 минут
st.subheader("🔮 Прогноз загруженности")
horizon = st.selectbox("Горизонт прогноза (минуты)", [30, 60, 120])

# Создаём будущие временные точки
last_minute_num = (last_time - min_time).total_seconds() / 60
future_minutes = np.arange(last_minute_num + 1, last_minute_num + horizon + 1)
future_times = [last_time + timedelta(minutes=i) for i in range(1, horizon+1)]

# Признаки для будущих точек: minute_num, hour, dayofweek
future_hours = [t.hour for t in future_times]
future_days = [t.weekday() for t in future_times]
X_future = pd.DataFrame({
    'minute_num': future_minutes,
    'hour': future_hours,
    'dayofweek': future_days
})
predictions = model.predict(X_future)
predictions = np.maximum(predictions, 0)  # неотрицательные

# Создаём DataFrame для прогноза
pred_df = pd.DataFrame({
    'minute_bucket': future_times,
    'total_vehicles': predictions,
    'type': 'прогноз'
})
hist_df = df[['minute_bucket', 'total_vehicles']].copy()
hist_df['type'] = 'история'
combined = pd.concat([hist_df, pred_df])

fig2 = px.line(combined, x='minute_bucket', y='total_vehicles', color='type',
               title=f"Прогноз на {horizon} минут вперед")
st.plotly_chart(fig2, use_container_width=True)

# 4. Сравнение с аналогичным периодом прошлого дня/недели (если есть данные)
st.subheader("📊 Сравнение с историей")
if len(df) > 1440:  # если больше суток данных
    # Последний час
    last_hour = df[df['minute_bucket'] >= last_time - timedelta(hours=1)]
    # Аналогичный час вчера
    yesterday_same = df[(df['minute_bucket'] >= last_time - timedelta(days=1, hours=1)) &
                        (df['minute_bucket'] < last_time - timedelta(days=1))]
    if not yesterday_same.empty:
        avg_now = last_hour['total_vehicles'].mean()
        avg_yest = yesterday_same['total_vehicles'].mean()
        st.metric("Средняя интенсивность за последний час", f"{avg_now:.1f}", delta=f"{avg_now - avg_yest:.1f} относительно вчера")

# 5. Структура потока (по типам ТС) - нужно из таблицы detections
@st.cache_data(ttl=60)
def load_types():
    conn = sqlite3.connect('data/traffic.db')
    df_types = pd.read_sql_query("SELECT vehicle_type, COUNT(*) as cnt FROM detections GROUP BY vehicle_type", conn)
    conn.close()
    return df_types

df_types = load_types()
if not df_types.empty:
    st.subheader("🚦 Структура транспортного потока")
    fig3 = px.pie(df_types, values='cnt', names='vehicle_type', title="Доля типов ТС")
    st.plotly_chart(fig3, use_container_width=True)

# 6. Доп. инсайты (опасные сближения и большегрузы) - упрощённо
st.subheader("⚠️ Системные инсайты")
# Для опасных сближений нужно анализировать координаты, упростим: покажем часы пик
peak_hours = df.groupby(df['minute_bucket'].dt.hour)['total_vehicles'].mean().sort_values(ascending=False).head(3)
st.write("🔹 Часы с максимальной загруженностью:", ", ".join([f"{int(h)}:00" for h in peak_hours.index]))
# Большегрузные (truck) - если есть
truck_count = df_types[df_types['vehicle_type']=='truck']['cnt'].values[0] if 'truck' in df_types['vehicle_type'].values else 0
st.write(f"🔹 Доля большегрузов: {truck_count/df_types['cnt'].sum()*100:.1f}%")

st.markdown("---")
st.caption("Данные обновляются автоматически при перезагрузке страницы")