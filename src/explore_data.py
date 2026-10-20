import duckdb

con = duckdb.connect("data/incident_intelligence.duckdb")

# 1. Load orders
con.execute("""
    CREATE OR REPLACE TABLE orders AS
    SELECT *
    FROM read_csv('data/raw/olist_orders_dataset.csv')
""")

# 2. Load order items
con.execute("""
    CREATE OR REPLACE TABLE order_items AS
    SELECT *
    FROM read_csv('data/raw/olist_order_items_dataset.csv')
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

con.close()


