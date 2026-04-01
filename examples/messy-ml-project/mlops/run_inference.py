# Databricks notebook source
"""Batch inference using the champion model from Unity Catalog.

Loads the model with the @champion alias, scores the input table,
and writes predictions to the output table. No code changes needed
in your project.
"""

import mlflow

# COMMAND ----------

dbutils.widgets.text("model_name", "")
dbutils.widgets.text("input_table_name", "")
dbutils.widgets.text("output_table_name", "")

model_name = dbutils.widgets.get("model_name")
input_table_name = dbutils.widgets.get("input_table_name")
output_table_name = dbutils.widgets.get("output_table_name")

# COMMAND ----------

# Load champion model
model = mlflow.pyfunc.load_model(f"models:/{model_name}@champion")
print(f"Loaded model: {model_name}@champion")

# COMMAND ----------

# Score input data
input_df = spark.table(input_table_name)
predictions = model.predict(input_df.toPandas())

# COMMAND ----------

# Write predictions
output_df = input_df.toPandas()
output_df["prediction"] = predictions
spark.createDataFrame(output_df).write.mode("overwrite").saveAsTable(output_table_name)
print(f"Wrote {len(output_df)} predictions to {output_table_name}")
