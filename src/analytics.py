import duckdb

def get_daily_metrics(date):
    con = duckdb.connect("data/incident_intelligence.duckdb")

    result = con.execute("""
        SELECT
            ROUND(SUM(i.price), 2) AS total_sales,
            COUNT(DISTINCT o.order_id) AS total_orders
        FROM orders AS o
        INNER JOIN order_items AS i
            ON o.order_id = i.order_id
        WHERE o.order_status = 'delivered'
          AND CAST(o.order_purchase_timestamp AS DATE) = ?
    """, [date]).fetchone() #single result compared to fetchall() which returns a list of tuples or show() which displays the result in a tabular format

    con.close()

    return result


if __name__ == "__main__":
    print(get_daily_metrics("2017-11-24"))