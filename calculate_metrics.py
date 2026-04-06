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