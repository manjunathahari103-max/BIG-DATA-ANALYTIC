from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum, avg, count, year, month
from pyspark.sql.types import DoubleType, IntegerType

# --------------------------------------
# 1. Create Spark Session
# --------------------------------------

spark = SparkSession.builder \
    .appName("Big Data Sales Analysis") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# --------------------------------------
# 2. Load Large Dataset
# --------------------------------------

df = spark.read.csv("large_sales_data.csv", 
                    header=True, 
                    inferSchema=True)

print("Total Records:", df.count())
df.printSchema()

# --------------------------------------
# 3. Data Cleaning
# --------------------------------------

# Drop null values
df = df.dropna()

# Cast columns
df = df.withColumn("quantity", col("quantity").cast(IntegerType()))
df = df.withColumn("price", col("price").cast(DoubleType()))

# Create total revenue column
df = df.withColumn("total_revenue", col("quantity") * col("price"))

# --------------------------------------
# 4. Basic Insights
# --------------------------------------

# 4.1 Total Revenue
total_revenue = df.agg(sum("total_revenue")).collect()[0][0]
print("Total Revenue:", total_revenue)

# 4.2 Top 5 Selling Products
top_products = df.groupBy("product") \
    .agg(sum("quantity").alias("total_quantity")) \
    .orderBy(col("total_quantity").desc()) \
    .limit(5)

print("Top 5 Products:")
top_products.show()

# 4.3 Revenue by Country
country_revenue = df.groupBy("country") \
    .agg(sum("total_revenue").alias("country_revenue")) \
    .orderBy(col("country_revenue").desc())

print("Revenue by Country:")
country_revenue.show()

# 4.4 Monthly Sales Trend
df = df.withColumn("year", year("order_date")) \
       .withColumn("month", month("order_date"))

monthly_sales = df.groupBy("year", "month") \
    .agg(sum("total_revenue").alias("monthly_revenue")) \
    .orderBy("year", "month")

print("Monthly Sales Trend:")
monthly_sales.show()

# --------------------------------------
# 5. Scalability Demonstration
# --------------------------------------

print("Number of Partitions:", df.rdd.getNumPartitions())

# Repartition for large scale optimization
df_repartitioned = df.repartition(8)

print("Partitions after Repartition:", df_repartitioned.rdd.getNumPartitions())

# --------------------------------------
# 6. Stop Spark Session
# --------------------------------------

spark.stop()
