
import duckdb
from collections import defaultdict, deque
from statistics import median

# Configuration
WINDOW_SIZE = 4
THRESHOLD = 30

# Connect to our existing database
con = duckdb.connect("data/incident_intelligence.duckdb")

# Retrieve daily sales
daily_sales = con.execute("""
    SELECT
        CAST(o.order_purchase_timestamp AS DATE) AS order_date,
        SUM(i.price) AS total_sales
    FROM orders AS o
    INNER JOIN order_items AS i
        ON o.order_id = i.order_id
    WHERE o.order_status = 'delivered'
    GROUP BY order_date
    ORDER BY order_date
""").fetchall()


con.close()



# Store the previous 4 values for each weekday
history = defaultdict(lambda: deque(maxlen=WINDOW_SIZE))

anomalies = []

for order_date, total_sales in daily_sales:
    weekday = order_date.weekday()
    actual = float(total_sales)

    previous_sales = history[weekday]

    # Wait until we have enough historical data
    if len(previous_sales) == WINDOW_SIZE:
        expected = median(previous_sales)

        if expected > 0:
            deviation = ((actual - expected) / expected) * 100

            if abs(deviation) >= THRESHOLD:
                anomalies.append(
                    (order_date, actual, expected, deviation)
                )

    # Add today's value only AFTER checking it
    previous_sales.append(actual)

print(f"Analyzed {len(daily_sales)} days")
print(f"Detected {len(anomalies)} anomalies\n")

for date, actual, expected, deviation in anomalies[:10]:
    print(
        f"{date} | Actual: {actual:.2f} | "
        f"Expected: {expected:.2f} | "
        f"Deviation: {deviation:+.2f}%"
    )

# Save detected anomalies to DuckDB
con = duckdb.connect("data/incident_intelligence.duckdb")

con.execute("""
    CREATE OR REPLACE TABLE detected_anomalies (
        order_date DATE,
        actual_sales DOUBLE,
        expected_sales DOUBLE,
        deviation_pct DOUBLE
    )
""")

con.executemany("""
    INSERT INTO detected_anomalies
    VALUES (?, ?, ?, ?)
""", anomalies)

con.close()

print("Anomalies saved to duckdb")