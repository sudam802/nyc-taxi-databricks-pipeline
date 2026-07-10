# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Gold Layer - Business Analytics
# MAGIC %md
# MAGIC # Gold Layer - Business Analytics
# MAGIC
# MAGIC This notebook creates business-ready aggregations and insights from the Silver layer data.
# MAGIC
# MAGIC **Analytics included:**
# MAGIC - Daily metrics (revenue, trips, averages)
# MAGIC - Peak hours analysis
# MAGIC - Popular routes
# MAGIC - Weekend vs Weekday patterns
# MAGIC - Time of day analysis
# MAGIC - Vendor performance comparison
# MAGIC - Advanced visualizations

# COMMAND ----------

# DBTITLE 1,Imports and Setup
from pyspark.sql.functions import col, sum, count, avg, min, max, hour, date_trunc, dayofweek, when
from pyspark.sql import functions as F

# Read Silver layer data
silver_df = spark.read.table("workspace.default.nyc_taxi_silver")

print(f"Total records in Silver layer: {silver_df.count():,}")
silver_df.printSchema()

# COMMAND ----------

# DBTITLE 1,1. Daily Metrics
# Daily aggregations
daily_metrics = silver_df.groupBy(
    F.date_trunc('day', 'tpep_pickup_datetime').alias('pickup_date')
).agg(
    F.count('*').alias('total_trips'),
    F.sum('total_amount').alias('total_revenue'),
    F.avg('fare_amount').alias('avg_fare'),
    F.avg('trip_distance').alias('avg_distance'),
    F.avg('duration_minutes').alias('avg_duration_min')
).orderBy('pickup_date')

print("Daily metrics sample:")
daily_metrics.show(10)

# Save to Gold table
daily_metrics.write.mode('overwrite').saveAsTable('workspace.default.nyc_taxi_gold_daily_metrics')
print("\nTable created: workspace.default.nyc_taxi_gold_daily_metrics")
daily_metrics

# COMMAND ----------

# DBTITLE 1,2. Peak Hours Analysis
# Hourly patterns
hourly_metrics = silver_df.withColumn('pickup_hour', F.hour('tpep_pickup_datetime')).groupBy('pickup_hour').agg(
    F.count('*').alias('total_trips'),
    F.sum('total_amount').alias('total_revenue'),
    F.avg('fare_amount').alias('avg_fare')
).orderBy('pickup_hour')

print("Peak hours analysis:")
hourly_metrics.show(24)

hourly_metrics.write.mode('overwrite').saveAsTable('workspace.default.nyc_taxi_gold_hourly_metrics')
print("\nTable created: workspace.default.nyc_taxi_gold_hourly_metrics")
hourly_metrics

# COMMAND ----------

# DBTITLE 1,3. Popular Routes
# Top pickup-dropoff pairs
popular_routes = silver_df.groupBy('PULocationID', 'DOLocationID').agg(
    F.count('*').alias('trip_count'),
    F.avg('fare_amount').alias('avg_fare'),
    F.avg('trip_distance').alias('avg_distance')
).orderBy(F.desc('trip_count')).limit(20)

print("Top 20 most popular routes:")
popular_routes.show()

popular_routes.write.mode('overwrite').saveAsTable('workspace.default.nyc_taxi_gold_popular_routes')
print("\nTable created: workspace.default.nyc_taxi_gold_popular_routes")
popular_routes

# COMMAND ----------

# DBTITLE 1,4. Weekend vs Weekday Patterns
# Weekend vs Weekday comparison
weekend_comparison = silver_df.withColumn(
    'is_weekend',
    when(F.dayofweek('tpep_pickup_datetime').isin([1, 7]), True).otherwise(False)
).groupBy('is_weekend').agg(
    F.count('*').alias('total_trips'),
    F.sum('total_amount').alias('total_revenue'),
    F.avg('fare_amount').alias('avg_fare'),
    F.avg('trip_distance').alias('avg_distance'),
    F.avg('tip_amount').alias('avg_tip')
)

print("Weekend vs Weekday analysis:")
weekend_comparison.show()

weekend_comparison.write.mode('overwrite').saveAsTable('workspace.default.nyc_taxi_gold_weekend_comparison')
print("\nTable created: workspace.default.nyc_taxi_gold_weekend_comparison")
weekend_comparison

# COMMAND ----------

# DBTITLE 1,5. Time of Day Analysis
# Time of day patterns
time_of_day_metrics = silver_df.withColumn(
    'time_of_day',
    when((F.hour('tpep_pickup_datetime') >= 6) & (F.hour('tpep_pickup_datetime') < 12), 'Morning')
    .when((F.hour('tpep_pickup_datetime') >= 12) & (F.hour('tpep_pickup_datetime') < 18), 'Afternoon')
    .when((F.hour('tpep_pickup_datetime') >= 18) & (F.hour('tpep_pickup_datetime') < 22), 'Evening')
    .otherwise('Night')
).groupBy('time_of_day').agg(
    F.count('*').alias('total_trips'),
    F.sum('total_amount').alias('total_revenue'),
    F.avg('fare_amount').alias('avg_fare'),
    F.avg('trip_distance').alias('avg_distance')
)

print("Time of day analysis:")
time_of_day_metrics.show()

time_of_day_metrics.write.mode('overwrite').saveAsTable('workspace.default.nyc_taxi_gold_time_of_day')
print("\nTable created: workspace.default.nyc_taxi_gold_time_of_day")
time_of_day_metrics

# COMMAND ----------

# DBTITLE 1,6. Vendor Performance Comparison
# Vendor comparison
vendor_metrics = silver_df.groupBy('VendorID').agg(
    F.count('*').alias('total_trips'),
    F.sum('total_amount').alias('total_revenue'),
    F.avg('fare_amount').alias('avg_fare'),
    F.avg('trip_distance').alias('avg_distance'),
    F.avg('tip_amount').alias('avg_tip')
).orderBy(F.desc('total_trips'))

print("Vendor performance comparison:")
vendor_metrics.show()

vendor_metrics.write.mode('overwrite').saveAsTable('workspace.default.nyc_taxi_gold_vendor_metrics')
print("\nTable created: workspace.default.nyc_taxi_gold_vendor_metrics")
vendor_metrics

# COMMAND ----------

# DBTITLE 1,7. Summary Statistics
# Overall summary
summary = silver_df.agg(
    F.count('*').alias('total_trips'),
    F.sum('total_amount').alias('total_revenue'),
    F.avg('fare_amount').alias('avg_fare'),
    F.avg('trip_distance').alias('avg_distance'),
    F.avg('duration_minutes').alias('avg_duration'),
    F.avg('tip_amount').alias('avg_tip'),
    F.min('tpep_pickup_datetime').alias('earliest_trip'),
    F.max('tpep_pickup_datetime').alias('latest_trip')
)

print("=" * 80)
print("NYC TAXI DATA - SUMMARY STATISTICS")
print("=" * 80)
summary.show(vertical=True)

summary.write.mode('overwrite').saveAsTable('workspace.default.nyc_taxi_gold_summary')
print("\nTable created: workspace.default.nyc_taxi_gold_summary")

# COMMAND ----------

# DBTITLE 1,8. Advanced Visualizations with Matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")

# Read Gold tables
hourly_metrics = spark.read.table("workspace.default.nyc_taxi_gold_hourly_metrics")
time_of_day_metrics = spark.read.table("workspace.default.nyc_taxi_gold_time_of_day")
weekend_comparison = spark.read.table("workspace.default.nyc_taxi_gold_weekend_comparison")
vendor_metrics = spark.read.table("workspace.default.nyc_taxi_gold_vendor_metrics")

# Create a figure with multiple subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Hourly patterns - Line plot with avg fare
hourly_data = hourly_metrics.toPandas()
ax1 = axes[0, 0]
ax1_twin = ax1.twinx()
ax1.plot(hourly_data['pickup_hour'], hourly_data['total_trips'], 'b-', linewidth=2, marker='o', label='Total Trips')
ax1_twin.plot(hourly_data['pickup_hour'], hourly_data['avg_fare'], 'r-', linewidth=2, marker='s', label='Avg Fare')
ax1.set_xlabel('Hour of Day', fontsize=12)
ax1.set_ylabel('Total Trips', color='b', fontsize=12)
ax1_twin.set_ylabel('Average Fare ($)', color='r', fontsize=12)
ax1.set_title('Hourly Trip Volume vs Average Fare', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper left')
ax1_twin.legend(loc='upper right')

# 2. Time of Day comparison - Grouped bar chart
time_data = time_of_day_metrics.toPandas()
ax2 = axes[0, 1]
x_pos = range(len(time_data))
width = 0.35
ax2.bar([p - width/2 for p in x_pos], time_data['total_trips']/1000, width, label='Trips (thousands)', color='skyblue')
ax2_twin = ax2.twinx()
ax2_twin.bar([p + width/2 for p in x_pos], time_data['avg_fare'], width, label='Avg Fare ($)', color='coral')
ax2.set_xlabel('Time of Day', fontsize=12)
ax2.set_ylabel('Total Trips (thousands)', fontsize=12)
ax2_twin.set_ylabel('Average Fare ($)', fontsize=12)
ax2.set_title('Trips and Fares by Time of Day', fontsize=14, fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(time_data['time_of_day'])
ax2.legend(loc='upper left')
ax2_twin.legend(loc='upper right')

# 3. Weekend vs Weekday comparison - Multiple metrics
weekend_data = weekend_comparison.toPandas()
weekend_data['is_weekend'] = weekend_data['is_weekend'].map({True: 'Weekend', False: 'Weekday'})
ax3 = axes[1, 0]
metrics = ['avg_fare', 'avg_distance', 'avg_tip']
x_pos = range(len(metrics))
weekday_vals = weekend_data[weekend_data['is_weekend']=='Weekday'][metrics].values[0]
weekend_vals = weekend_data[weekend_data['is_weekend']=='Weekend'][metrics].values[0]
ax3.bar([p - width/2 for p in x_pos], weekday_vals, width, label='Weekday', color='steelblue')
ax3.bar([p + width/2 for p in x_pos], weekend_vals, width, label='Weekend', color='orange')
ax3.set_ylabel('Value ($)', fontsize=12)
ax3.set_title('Weekend vs Weekday - Key Metrics Comparison', fontsize=14, fontweight='bold')
ax3.set_xticks(x_pos)
ax3.set_xticklabels(['Avg Fare', 'Avg Distance (mi)', 'Avg Tip'])
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

# 4. Vendor performance comparison
vendor_data = vendor_metrics.toPandas()
ax4 = axes[1, 1]
ax4.bar(vendor_data['VendorID'].astype(str), vendor_data['total_revenue']/1000000, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#BC4B51'][:len(vendor_data)])
ax4.set_xlabel('Vendor ID', fontsize=12)
ax4.set_ylabel('Total Revenue ($ Millions)', fontsize=12)
ax4.set_title('Revenue by Vendor', fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')

for i, (vendor, revenue, trips) in enumerate(zip(vendor_data['VendorID'], vendor_data['total_revenue'], vendor_data['total_trips'])):
    ax4.text(i, revenue/1000000 + 0.5, f'{trips:,} trips', ha='center', fontsize=9)

plt.tight_layout()
plt.show()

print("Advanced visualizations created!")
print("\nKey Insights:")
print(f"- Busiest hour: {hourly_data.loc[hourly_data['total_trips'].idxmax(), 'pickup_hour']}:00 with {hourly_data['total_trips'].max():,} trips")
print(f"- Highest avg fare hour: {hourly_data.loc[hourly_data['avg_fare'].idxmax(), 'pickup_hour']}:00 at ${hourly_data['avg_fare'].max():.2f}")
print(f"- Most profitable time: {time_data.loc[time_data['total_revenue'].idxmax(), 'time_of_day']} with ${time_data['total_revenue'].max():,.2f}")

# COMMAND ----------


