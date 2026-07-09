# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Read from Bronze Table
from pyspark.sql.functions import col, unix_timestamp, round as spark_round, when, hour, dayofweek, to_date
from pyspark.sql import Window

# Read from Bronze table
bronze_df = spark.read.table("workspace.default.nyc_taxi_bronze")

print(f"📊 Bronze records loaded: {bronze_df.count():,}")
print(f"📋 Columns: {len(bronze_df.columns)}")

# COMMAND ----------

# DBTITLE 1,Remove Duplicates
# Remove duplicate records based on key columns
# From your analysis: VendorID, pickup/dropoff times, and location IDs identify unique trips

key_cols = ['VendorID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime', 'PULocationID', 'DOLocationID']

# Drop exact duplicates
deduped_df = bronze_df.dropDuplicates(key_cols)

duplicates_removed = bronze_df.count() - deduped_df.count()
print(f"🗑️ Duplicates removed: {duplicates_removed:,}")
print(f"✅ Remaining records: {deduped_df.count():,}")

# COMMAND ----------

# DBTITLE 1,Handle Null Values
# Check null counts before cleaning
from pyspark.sql.functions import sum as spark_sum

null_counts_before = deduped_df.select([
    spark_sum(col(c).isNull().cast("int")).alias(c)
    for c in deduped_df.columns
]).collect()[0].asDict()

print("📊 Null counts by column:")
for col_name, null_count in null_counts_before.items():
    if null_count > 0:
        print(f"  {col_name}: {null_count:,}")

# Remove records with critical nulls (pickup/dropoff times, locations, amounts)
critical_columns = ['tpep_pickup_datetime', 'tpep_dropoff_datetime', 'PULocationID', 'DOLocationID', 
                    'fare_amount', 'total_amount', 'trip_distance']

no_nulls_df = deduped_df.dropna(subset=critical_columns)

nulls_removed = deduped_df.count() - no_nulls_df.count()
print(f"\n🗑️ Records with critical nulls removed: {nulls_removed:,}")
print(f"✅ Remaining records: {no_nulls_df.count():,}")

# COMMAND ----------

# DBTITLE 1,Filter Invalid Records
# Remove invalid records based on your analysis:
# 1. Negative monetary values (except VendorID=2 which are refunds)
# 2. Zero or negative trip distances
# 3. Invalid pickup/dropoff times (dropoff before pickup)

valid_df = no_nulls_df.filter(
    # Remove negative amounts (except VendorID=2 refunds)
    ((col("fare_amount") >= 0) | (col("VendorID") == 2)) &
    ((col("tip_amount") >= 0) | (col("VendorID") == 2)) &
    ((col("total_amount") >= 0) | (col("VendorID") == 2)) &
    # Valid trip distance
    (col("trip_distance") > 0) &
    # Dropoff must be after pickup
    (col("tpep_dropoff_datetime") > col("tpep_pickup_datetime"))
)

invalid_removed = no_nulls_df.count() - valid_df.count()
print(f"🗑️ Invalid records removed: {invalid_removed:,}")
print(f"✅ Valid records remaining: {valid_df.count():,}")

# COMMAND ----------

# DBTITLE 1,Add Calculated Columns
# Add useful calculated columns for analysis

silver_df = valid_df \
    .withColumn(
        "trip_duration_minutes",
        spark_round(
            (unix_timestamp("tpep_dropoff_datetime") - unix_timestamp("tpep_pickup_datetime")) / 60,
            2
        )
    ) \
    .withColumn(
        "average_speed_mph",
        spark_round(
            when(col("trip_duration_minutes") > 0,
                 col("trip_distance") / (col("trip_duration_minutes") / 60)
            ).otherwise(0),
            2
        )
    ) \
    .withColumn(
        "pickup_hour",
        hour("tpep_pickup_datetime")
    ) \
    .withColumn(
        "pickup_day_of_week",
        dayofweek("tpep_pickup_datetime")  # 1=Sunday, 7=Saturday
    ) \
    .withColumn(
        "pickup_date",
        to_date("tpep_pickup_datetime")
    ) \
    .withColumn(
        "time_of_day",
        when((col("pickup_hour") >= 6) & (col("pickup_hour") < 12), "Morning")
        .when((col("pickup_hour") >= 12) & (col("pickup_hour") < 17), "Afternoon")
        .when((col("pickup_hour") >= 17) & (col("pickup_hour") < 21), "Evening")
        .otherwise("Night")
    ) \
    .withColumn(
        "is_weekend",
        when(col("pickup_day_of_week").isin([1, 7]), True).otherwise(False)
    )

print("✅ Calculated columns added:")
print("  - trip_duration_minutes")
print("  - average_speed_mph")
print("  - pickup_hour, pickup_day_of_week, pickup_date")
print("  - time_of_day (Morning/Afternoon/Evening/Night)")
print("  - is_weekend (True/False)")

# Show sample
display(silver_df.select(
    "VendorID", "pickup_date", "trip_distance", "trip_duration_minutes", 
    "average_speed_mph", "time_of_day", "is_weekend", "total_amount"
).limit(10))

# COMMAND ----------

# DBTITLE 1,Data Quality Checks
# Perform data quality checks on the Silver data

# Check for unrealistic values
quality_issues = silver_df.filter(
    (col("trip_duration_minutes") <= 0) |
    (col("trip_duration_minutes") > 1440) |  # > 24 hours
    (col("average_speed_mph") > 100) |  # Unrealistic speed for NYC
    (col("trip_distance") > 100)  # Unrealistic distance
)

print("📊 Data Quality Report:")
print(f"  Total Silver records: {silver_df.count():,}")
print(f"  Potential quality issues: {quality_issues.count():,}")

if quality_issues.count() > 0:
    print("\n⚠️ Sample quality issues:")
    display(quality_issues.select(
        "trip_distance", "trip_duration_minutes", "average_speed_mph", "total_amount"
    ).limit(10))
    
    # Optional: filter out extreme outliers
    # silver_df = silver_df.filter(
    #     (col("trip_duration_minutes") > 0) &
    #     (col("trip_duration_minutes") <= 1440) &
    #     (col("average_speed_mph") <= 100)
    # )

# COMMAND ----------

# DBTITLE 1,Write to Silver Delta Table
# Write the cleaned and enriched data to Silver Delta table

silver_df.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("workspace.default.nyc_taxi_silver")

print("\n✅ Silver table created: workspace.default.nyc_taxi_silver")
print(f"📊 Final record count: {silver_df.count():,}")
print("\n🎯 Silver layer complete! Data is now clean, validated, and enriched.")
print("\n📈 Ready for Gold layer aggregations and analytics!")

# COMMAND ----------


