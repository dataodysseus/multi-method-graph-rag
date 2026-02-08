# Databricks notebook source
# MAGIC %md
# MAGIC ###Install Library

# COMMAND ----------

# MAGIC %pip install prophet

# COMMAND ----------

# dbutils.widgets.text("catalog", "")
# dbutils.widgets.text("schema", "")

# COMMAND ----------

# MAGIC %md
# MAGIC ###Load inputs

# COMMAND ----------

inputCatalog = dbutils.widgets.get("catalog")
inpuSchema = dbutils.widgets.get("schema")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Import libraries & set log level

# COMMAND ----------

import pandas as pd
import datetime
from prophet import Prophet
import logging

logging.getLogger("py4j").setLevel(logging.ERROR)

# COMMAND ----------

# MAGIC %md
# MAGIC ###Load source data
# MAGIC * Provides items sold by store by date

# COMMAND ----------

df = spark.table(f'{inputCatalog}.{inpuSchema}.items_sales')
df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC * Aggregated sales table at location level 

# COMMAND ----------

# DBTITLE 1,Cell 10
from pyspark.sql.functions import sum as _sum

df_forecast = df.groupBy('location_id','sale_date').agg(_sum('total_sales_value').alias('total_sales')).orderBy('location_id','sale_date')
df_forecast.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ###Create forecasts by sotres and by items 
# MAGIC * Iterate through each row and identify unused store-item combinations (use each combination only once)
# MAGIC * Convert filtered data to Pandas (this way whole dataset, can contain billions of records, is not converted to Pandas)
# MAGIC * Create input dataset for Prophet, prophet_df
# MAGIC * Calculate `periods` based on latest dataset date and target days of forecastig 
# MAGIC * Run model and create output dataset
# MAGIC * Convert output Pandas dataframe to Spark dataframe
# MAGIC * Overwrite using first dataframe, append subsequent ones, this will ensure efficient update of the final dataset

# COMMAND ----------

import calendar
from datetime import date

todays_date = pd.to_datetime("2026-01-01")
forecast_run_date = todays_date - pd.Timedelta(days=1)
last_day = calendar.monthrange(todays_date.year, todays_date.month)[1]
last_day_of_month = todays_date.replace(day=last_day)
print(todays_date, forecast_run_date, last_day_of_month)

# COMMAND ----------

# DBTITLE 1,Untitled
data_itr = df.rdd.toLocalIterator()

storeItemList = []

counter = 1
 
for row in data_itr:
    storeLocation = row['location_id']
    if(storeLocation not in storeItemList):    
        storeItemList.append(storeLocation)
        print(storeLocation, storeItemList)
        if (storeLocation in storeItemList):
            sdf = df_forecast.filter((df.location_id == row['location_id']) & (df.sale_date <= todays_date))

            pdf = sdf.select('sale_date','total_sales').toPandas()
            pdf['date']=pd.to_datetime(pdf['sale_date'])

            max_data_date = pdf['date'].max()
            target_date = last_day_of_month
            foreperiods = (target_date - max_data_date).days
            print(max_data_date, target_date, foreperiods)

            prophet_df = pd.DataFrame()
            prophet_df["ds"] = pd.to_datetime(pdf["date"])
            prophet_df["y"] = pdf["total_sales"]
            prophet_df = prophet_df.dropna()

            holidays = pd.DataFrame({"ds": [], "holiday": []})
            prophet_obj = Prophet(holidays=holidays)
            prophet_obj.add_country_holidays(country_name='US')
            prophet_obj.fit(prophet_df)

            prophet_future = prophet_obj.make_future_dataframe(periods=foreperiods, freq = "d")
            prophet_forecast = prophet_obj.predict(prophet_future)

            result_df = prophet_forecast[['ds', 'yhat']][prophet_forecast['ds']>=todays_date]
            result_df = result_df.rename(columns={'ds': 'forecast_date', 'yhat': 'forecast_sales_value'})
            result_df['location_id'] = row['location_id']
            result_df[['forecast_sales_value']] = result_df[['forecast_sales_value']].apply(lambda x: pd.Series.round(x, 2))
            result_df['forecast_run_date'] = forecast_run_date

            if (counter == 0):
                final_df = spark.createDataFrame(result_df)
                final_df.write.mode('overwrite').saveAsTable(f'{inputCatalog}.{inpuSchema}.items_sales_forecast')
            if (counter > 0):
                final_df = spark.createDataFrame(result_df)
                final_df.write.mode("append").saveAsTable(f'{inputCatalog}.{inpuSchema}.items_sales_forecast')

            counter += 1

# COMMAND ----------

result_df.tail(50)

# COMMAND ----------

output_df = spark.table(f'{inputCatalog}.{inpuSchema}.items_sales_forecast')
output_df.display()

# COMMAND ----------

df_forecast.display()

# COMMAND ----------

# DBTITLE 1,Cell 17
store_agg_df = (output_df
                    .join(df_forecast, (output_df.location_id == df_forecast.location_id) & (output_df.forecast_date == df_forecast.sale_date), how="left")
                    .drop(df_forecast.location_id))
store_agg_df.display()

# COMMAND ----------

store_agg_df.write.mode('overwrite').saveAsTable(f'{inputCatalog}.{inpuSchema}.items_sales_aggregated_stores')

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM $catalog.$schema.items_sales_aggregated_stores LIMIT 10;
