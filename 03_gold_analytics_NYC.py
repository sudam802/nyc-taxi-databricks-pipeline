# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Read Silver Table
from pyspark.sql.functions import col, sum, avg, count, round as spark_round, desc, when

# Read from Silver table
silver_df = spark.read.table("workspace.default.nyc_taxi_silver")

print(f"Silver records loaded: {silver_df.count():,}")
print(f"Date range: {silver_df.agg({'pickup_date': 'min'}).collect()[0][0]} to {silver_df.agg({'pickup_date': 'max'}).collect()[0][0]}")

# COMMAND ----------

# DBTITLE 1,Daily Revenue and Trip Metrics
# Aggregate daily metrics
daily_metrics = silver_df.groupBy("pickup_date") \
    .agg(
        count("*").alias("total_trips"),
        spark_round(sum("total_amount"), 2).alias("total_revenue"),
        spark_round(avg("total_amount"), 2).alias("avg_fare"),
        spark_round(avg("trip_distance"), 2).alias("avg_distance"),
        spark_round(avg("trip_duration_minutes"), 2).alias("avg_duration_min")
    ) \
    .orderBy("pickup_date")

print("Daily metrics sample:")
display(daily_metrics.limit(10))

# Write to Gold table
daily_metrics.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("workspace.default.nyc_taxi_gold_daily_metrics")

print("\nTable created: workspace.default.nyc_taxi_gold_daily_metrics")

# COMMAND ----------

# DBTITLE 1,Peak Hours Analysis
# Aggregate by hour of day
hourly_metrics = silver_df.groupBy("pickup_hour") \
    .agg(
        count("*").alias("total_trips"),
        spark_round(sum("total_amount"), 2).alias("total_revenue"),
        spark_round(avg("total_amount"), 2).alias("avg_fare")
    ) \
    .orderBy("pickup_hour")

print("Peak hours analysis:")
display(hourly_metrics)

# Write to Gold table
hourly_metrics.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("workspace.default.nyc_taxi_gold_hourly_metrics")

print("\nTable created: workspace.default.nyc_taxi_gold_hourly_metrics")

# COMMAND ----------

# DBTITLE 1,Popular Routes Analysis
# Find top 20 routes by trip count
popular_routes = silver_df.groupBy("PULocationID", "DOLocationID") \
    .agg(
        count("*").alias("trip_count"),
        spark_round(avg("total_amount"), 2).alias("avg_fare"),
        spark_round(avg("trip_distance"), 2).alias("avg_distance"),
        spark_round(avg("trip_duration_minutes"), 2).alias("avg_duration_min")
    ) \
    .orderBy(desc("trip_count")) \
    .limit(20)

print("Top 20 popular routes:")
display(popular_routes)

# Write to Gold table
popular_routes.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("workspace.default.nyc_taxi_gold_popular_routes")

print("\nTable created: workspace.default.nyc_taxi_gold_popular_routes")

# COMMAND ----------

# DBTITLE 1,Weekend vs Weekday Comparison
# Compare weekend vs weekday patterns
weekend_comparison = silver_df.groupBy("is_weekend") \
    .agg(
        count("*").alias("total_trips"),
        spark_round(sum("total_amount"), 2).alias("total_revenue"),
        spark_round(avg("total_amount"), 2).alias("avg_fare"),
        spark_round(avg("trip_distance"), 2).alias("avg_distance"),
        spark_round(avg("trip_duration_minutes"), 2).alias("avg_duration_min"),
        spark_round(avg("tip_amount"), 2).alias("avg_tip")
    )

print("Weekend vs Weekday comparison:")
display(weekend_comparison)

# Write to Gold table
weekend_comparison.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("workspace.default.nyc_taxi_gold_weekend_comparison")

print("\nTable created: workspace.default.nyc_taxi_gold_weekend_comparison")

# COMMAND ----------

# DBTITLE 1,Time of Day Analysis
# Analyze by time of day (Morning/Afternoon/Evening/Night)
time_of_day_metrics = silver_df.groupBy("time_of_day") \
    .agg(
        count("*").alias("total_trips"),
        spark_round(sum("total_amount"), 2).alias("total_revenue"),
        spark_round(avg("total_amount"), 2).alias("avg_fare"),
        spark_round(avg("trip_distance"), 2).alias("avg_distance")
    ) \
    .orderBy(
        when(col("time_of_day") == "Morning", 1)
        .when(col("time_of_day") == "Afternoon", 2)
        .when(col("time_of_day") == "Evening", 3)
        .otherwise(4)
    )

print("Time of day analysis:")
display(time_of_day_metrics)

# Write to Gold table
time_of_day_metrics.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("workspace.default.nyc_taxi_gold_time_of_day")

print("\nTable created: workspace.default.nyc_taxi_gold_time_of_day")

# COMMAND ----------

# DBTITLE 1,Vendor Performance Comparison
# Compare vendor performance
vendor_metrics = silver_df.groupBy("VendorID") \
    .agg(
        count("*").alias("total_trips"),
        spark_round(sum("total_amount"), 2).alias("total_revenue"),
        spark_round(avg("total_amount"), 2).alias("avg_fare"),
        spark_round(avg("trip_distance"), 2).alias("avg_distance"),
        spark_round(avg("trip_duration_minutes"), 2).alias("avg_duration_min"),
        spark_round(avg("average_speed_mph"), 2).alias("avg_speed_mph")
    ) \
    .orderBy("VendorID")

print("Vendor performance comparison:")
display(vendor_metrics)

# Write to Gold table
vendor_metrics.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("workspace.default.nyc_taxi_gold_vendor_metrics")

print("\nTable created: workspace.default.nyc_taxi_gold_vendor_metrics")

# COMMAND ----------

# DBTITLE 1,Summary Statistics
# Overall summary statistics
print("Gold Layer Complete!")
print("\nCreated Gold tables:")
print("1. workspace.default.nyc_taxi_gold_daily_metrics - Daily revenue and trip metrics")
print("2. workspace.default.nyc_taxi_gold_hourly_metrics - Peak hours analysis")
print("3. workspace.default.nyc_taxi_gold_popular_routes - Top 20 routes")
print("4. workspace.default.nyc_taxi_gold_weekend_comparison - Weekend vs weekday patterns")
print("5. workspace.default.nyc_taxi_gold_time_of_day - Morning/Afternoon/Evening/Night analysis")
print("6. workspace.default.nyc_taxi_gold_vendor_metrics - Vendor performance comparison")
print("\nThese tables are optimized for dashboards and reporting!")

# COMMAND ----------


