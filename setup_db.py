import sqlite3
import os

# Убедимся, что папка data существует
os.makedirs('data', exist_ok=True)

# Подключаемся к базе (файл создастся автоматически)
conn = sqlite3.connect('data/traffic.db')
cursor = conn.cursor()

# Создаём таблицу для хранения событий
cursor.execute('''
    CREATE TABLE IF NOT EXISTS detections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,           -- Время детекции
        vehicle_type TEXT,        -- Тип ТС (car, bus, truck)
        confidence REAL,          -- Уверенность модели (0..1)
        x REAL, y REAL, w REAL, h REAL,  -- Координаты bounding box
        frame_id INTEGER          -- Номер кадра
    )
''')

# Создаём таблицу для агрегированных метрик (для дашборда)
cursor.execute('''
    CREATE TABLE IF NOT EXISTS metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        minute_bucket TEXT,       -- Минута (2024-01-01 12:00:00)
        total_vehicles INTEGER,
        avg_confidence REAL
    )
''')

conn.commit()
conn.close()
print("База данных и таблицы созданы!")