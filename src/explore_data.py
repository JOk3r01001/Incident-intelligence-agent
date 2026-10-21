import duckdb

con = duckdb.connect("data/incident_intelligence.duckdb")

# 1. Load orders
con.execute("""
    CREATE OR REPLACE TABLE orders AS
    SELECT *
    FROM read_csv('data/olist_orders_dataset.csv')
""")

# 2. Load order items
con.execute("""
    CREATE OR REPLACE TABLE order_items AS
    SELECT *
    FROM read_csv('data/olist_order_items_dataset.csv')
""")

# 3. Explore orders (our previous queries)
print("TOTAL ORDERS:")
con.sql("SELECT COUNT(*) FROM orders").show()

print("\nORDERS BY STATUS:")
con.sql("""
    SELECT order_status, COUNT(*) AS total_orders
    FROM orders
    GROUP BY order_status
    ORDER BY total_orders DESC
""").show()

# 4. Explore relationships using JOIN
print("\nJOINED DATA:")
con.sql("""
    SELECT
        o.order_id,
        o.order_status,
        i.product_id,
        i.price
    FROM orders AS o
    INNER JOIN order_items AS i
        ON o.order_id = i.order_id
    LIMIT 10
""").show()

print("\nDAILY SALES ANALYTICS:")


# 5. Daily sales analytics
con.sql("""
    SELECT
        CAST(o.order_purchase_timestamp AS DATE) AS order_date,
        SUM(i.price) AS total_sales,
        COUNT(DISTINCT o.order_id) AS total_orders
    FROM orders AS o
    INNER JOIN order_items AS i
        ON o.order_id = i.order_id
    GROUP BY order_date
    ORDER BY order_date
    LIMIT 15
""").show()

# 6. Investigate January 2017 sales
print("\nJANUARY 2017 SALES INVESTIGATION:")

con.sql("""
    SELECT
        CAST(o.order_purchase_timestamp AS DATE) AS order_date,
        SUM(i.price) AS total_sales,
        COUNT(DISTINCT o.order_id) AS total_orders
    FROM orders AS o
    INNER JOIN order_items AS i
        ON o.order_id = i.order_id
    WHERE o.order_status = 'delivered'
      AND CAST(o.order_purchase_timestamp AS DATE)
          BETWEEN '2017-01-18' AND '2017-02-05'
    GROUP BY order_date
    ORDER BY order_date
""").show()

# 7. Investigate anomalies in daily sales
print("\nDETECTED ANOMALIES:")

con.sql("""
    SELECT *
    FROM detected_anomalies
    ORDER BY ABS(deviation_pct) DESC
    LIMIT 20;
""").show()



# 8. Investigate Black Friday sales  
print("\nBLACK FRIDAY INVESTIGATION:")

con.sql("""
    SELECT
        CAST(o.order_purchase_timestamp AS DATE) AS order_date,
        ROUND(SUM(i.price), 2) AS total_sales,
        COUNT(DISTINCT o.order_id) AS total_orders,
        ROUND(
            SUM(i.price) / COUNT(DISTINCT o.order_id),
            2
        ) AS average_order_value
    FROM orders AS o
    INNER JOIN order_items AS i
        ON o.order_id = i.order_id
    WHERE o.order_status = 'delivered'
      AND CAST(o.order_purchase_timestamp AS DATE)
          BETWEEN '2017-11-17' AND '2017-11-30'
    GROUP BY order_date
    ORDER BY order_date
""").show()




con.close()


