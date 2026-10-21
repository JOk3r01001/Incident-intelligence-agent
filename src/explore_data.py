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

# 3. Load products
con.execute("""
    CREATE OR REPLACE TABLE products AS
    SELECT *
    FROM read_csv('data/olist_products_dataset.csv')
""")


# 4. Explore orders
print("TOTAL ORDERS:")
con.sql("SELECT COUNT(*) FROM orders").show()

print("\nORDERS BY STATUS:")
con.sql("""
    SELECT order_status, COUNT(*) AS total_orders
    FROM orders
    GROUP BY order_status
    ORDER BY total_orders DESC
""").show()

# 5. Explore relationships using JOIN
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


# 6. Daily sales analytics
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

# 7. Investigate January 2017 sales
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

# 8. Investigate anomalies in daily sales
print("\nDETECTED ANOMALIES:")

con.sql("""
    SELECT *
    FROM detected_anomalies
    ORDER BY ABS(deviation_pct) DESC
    LIMIT 20;
""").show()



# 9. Investigate Black Friday sales  
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


# 11. Show black friday sales by category
print("\nBLACK FRIDAY SALES BY CATEGORY:")

con.sql("""
    SELECT
        p.product_category_name,
        ROUND(SUM(i.price), 2) AS total_sales,
        COUNT(DISTINCT o.order_id) AS total_orders
    FROM orders AS o
    INNER JOIN order_items AS i
        ON o.order_id = i.order_id
    INNER JOIN products AS p
        ON i.product_id = p.product_id
    WHERE o.order_status = 'delivered'
      AND CAST(o.order_purchase_timestamp AS DATE) = '2017-11-24'
    GROUP BY p.product_category_name
    ORDER BY total_sales DESC
    LIMIT 10
""").show()


# 12. Show all tables in the database
print("\nTABLES IN THE DATABASE:")
con.sql("SHOW TABLES").show()

# 13. Investigate product categories
print("\nPRODUCT CATEGORY INVESTIGATION:")

con.sql("""
    SELECT
        o.order_id,
        i.price,
        p.product_category_name
    FROM orders AS o
    INNER JOIN order_items AS i
        ON o.order_id = i.order_id
    INNER JOIN products AS p
        ON i.product_id = p.product_id
    LIMIT 10
""").show()


con.close()


