import cv2
from ultralytics import YOLO
import sqlite3
from datetime import datetime
import os
import sys
import pandas as pd

# ------------------------------------------------------------
# 1. СОЗДАЁМ БАЗУ И ТАБЛИЦЫ (если их нет)
# ------------------------------------------------------------
def setup_database():
    os.makedirs('data', exist_ok=True)
    conn = sqlite3.connect('data/traffic.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            vehicle_type TEXT,
            confidence REAL,
            x REAL, y REAL, w REAL, h REAL,
            frame_id INTEGER
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            minute_bucket TEXT,
            total_vehicles INTEGER,
            avg_confidence REAL
        )
    ''')
    conn.commit()
    conn.close()
    print("✅ База данных и таблицы готовы")

# ------------------------------------------------------------
# 2. ДЕТЕКЦИЯ (обработка видео)
# ------------------------------------------------------------
def run_detection(video_path='data/traffic_video.mp4'):
    if not os.path.exists(video_path):
        print(f"❌ Ошибка: файл {video_path} не найден!")
        sys.exit(1)

    print("Загружаем модель YOLOv8n...")
    model = YOLO('yolov8n.pt')
    print("✅ Модель загружена")

    conn = sqlite3.connect('data/traffic.db')
    cursor = conn.cursor()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Не удалось открыть видео {video_path}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Видео: {total_frames} кадров")
    frame_id = 0
    detections_count = 0

    print("Начинаем детекцию...")
    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model(frame, verbose=False)
        for box in results[0].boxes:
            if int(box.cls) in [2, 5, 7]:
                vehicle_type = model.names[int(box.cls)]
                confidence = float(box.conf[0])
                x, y, w, h = box.xywh[0].tolist()
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                cursor.execute('''
                    INSERT INTO detections (timestamp, vehicle_type, confidence, x, y, w, h, frame_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (timestamp, vehicle_type, confidence, x, y, w, h, frame_id))
                detections_count += 1

        if frame_id % 50 == 0:
            conn.commit()
            print(f"Кадров: {frame_id}/{total_frames}, найдено: {detections_count}")
        frame_id += 1

    cap.release()
    conn.commit()
    conn.close()
    print(f"✅ Детекция завершена. Найдено {detections_count} транспортных средств.")

# ------------------------------------------------------------
# 3. РАСЧЁТ МЕТРИК (агрегация по минутам)
# ------------------------------------------------------------
def calculate_metrics():
    conn = sqlite3.connect('data/traffic.db')
    df = pd.read_sql_query("SELECT id, timestamp, vehicle_type, confidence FROM detections", conn)
    if df.empty:
        print("❌ Нет данных для агрегации. Детекция не нашла ни одного объекта.")
        conn.close()
        return

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['minute_bucket'] = df['timestamp'].dt.floor('min')
    metrics = df.groupby('minute_bucket').agg(
        total_vehicles=('id', 'count'),
        avg_confidence=('confidence', 'mean')
    ).reset_index()

    cursor = conn.cursor()
    cursor.execute("DELETE FROM metrics")
    for _, row in metrics.iterrows():
        cursor.execute('''
    INSERT INTO metrics (minute_bucket, total_vehicles, avg_confidence)
    VALUES (?, ?, ?)
''', (str(row['minute_bucket']), row['total_vehicles'], row['avg_confidence']))
    conn.commit()
    conn.close()
    print(f"✅ Метрики рассчитаны: {len(metrics)} временных интервалов")
    print(metrics.head())

# ------------------------------------------------------------
# 4. ЗАПУСК ВСЕГО ПАЙПЛАЙНА
# ------------------------------------------------------------
if __name__ == "__main__":
    setup_database()
    run_detection()
    calculate_metrics()
    print("\n🎉 ВСЁ ГОТОВО! Данные в data/traffic.db можно использовать для дашборда.")