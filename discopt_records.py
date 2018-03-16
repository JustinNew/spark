#########################################################################################################################################
#
#  Discount Tree Optimization Spark Code
#  This code takes in account_key, var, val, sales and discount as input.
#  It builds a decision tree (or random forest) for every discount level to predict sales.
#  For each account_key, sales for all possible discount levels are predicted.
#  The output of the script is a driver table written into Hive.
#  The driver table includes account_key, discount, and associated sales.
#
#########################################################################################################################################

from __future__ import print_function

from pyspark.sql import SQLContext, HiveContext, SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import DecisionTreeRegressor, RandomForestRegressor
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.stat import Statistics

import sys, re

# Run the program as:
command_line = "nohup spark-submit --queue adhoc --executor-memory 40G --num-executors 6 discopt_records.py 'DTO_Tree' 'db_sas_discount' 'ce_20180302_train' 'ce_20180302_score' 'true' >& discopt_records.out &"

# Command line parameters.
if len(sys.argv) == 6:

    model_name = sys.argv[1]
    database_name = sys.argv[2]
    table_train = sys.argv[3]
    table_score = sys.argv[4]
    flag_score = sys.argv[5]

else:

    print("Using the default parameters." + command_line)
    model_name = 'DTO_Tree'
    database_name = 'db_sas_discount'
    table_train = 'ce_20180302_train'   
    table_score = 'ce_20180302_score'
    flag_score = 'true'

#########################################################################
# Create a spark Session.

spark = SparkSession.builder.appName("Discount Tree").enableHiveSupport().getOrCreate()

sc = spark.sparkContext
sc.setLogLevel('FATAL')

sqlContext = SQLContext(sc)

#########################################################################
#
# Query to retrieve the customer data incl. account_key, var, val and trgt.
#
#########################################################################
if flag_score == "true":
    t_score = "SCORE"
else:
    t_score = "TRAIN"
print("#########################################################################")
print("Running Spark Code to Create Discount Records.")
print("The output is the records for all account_keys in ", t_score, " dataset.") 
print("#########################################################################")

# There are about 25 M account_keys in the dataset. 
# There are three kinds of records: Discount = 15/20/30. 
# Get the VARs and associated VALs for each of the Account_Key.
########################################################################################
# Get TRAIN DATA.
sqlStatementInputs1 = "select * from " + database_name + "." + table_train + "_vars"

# Retrieve VARs and VALs data.
df1 = sqlContext.sql(sqlStatementInputs1).groupBy("account_key").pivot("var").agg({"val": "sum"})

# Get the Discount for each Account_Key.
sqlStatementInputs2 = "select account_key, discount from " + database_name + "." + table_train

# Retrieve Discount Data.
df2 = sqlContext.sql(sqlStatementInputs2)   

# Get TRGT for each Account_Key.
sqlStatementInputs3 = "select account_key, trgt from " + database_name + "." + table_train + "_trgt"

# Retrieves TRGT Data
df3 = sqlContext.sql(sqlStatementInputs3)   

# Join all three DataFrames and get them the correct data types.
df = df1.join(df2,["account_key"])
df = df.join(df3, ["account_key"])

# Casting columns.
condition = lambda col: 'F_' in col
fcols = df.select(*filter(condition,df.columns)).columns
for colname in fcols:
    df = df.withColumn("temp", df[colname].cast("int")).drop(colname).withColumnRenamed("temp", colname)

condition = lambda col: 'R_' in col
rcols = df.select(*filter(condition,df.columns)).columns
for colname in rcols:
    df = df.withColumn("temp", df[colname].cast("int")).drop(colname).withColumnRenamed("temp", colname)
    
condition = lambda col: 'M_' in col
mcols = df.select(*filter(condition,df.columns)).columns
for colname in mcols:
    df = df.withColumn("temp", df[colname].cast("float")).drop(colname).withColumnRenamed("temp", colname)
    
dcols = ["discount", "AGE", "ANNIV", "ANNIV_DISC", "FEMALE", "INCOME", "KC_OPEN", "MALE", "MARRIED", "POC", "PREVIOUS_DISCOUNT"]
for colname in dcols:
    df = df.withColumn("temp", df[colname].cast("float")).drop(colname).withColumnRenamed("temp", colname)
    
df = df.withColumn("temp", df.trgt.cast("float")).drop("trgt").withColumnRenamed("temp", "trgt")
########################################################################################
# Done Get TRAIN DATA.

########################################################################################
# Get SCORE DATA.
# Get the VARs and associated VALs for each of the Account_Key.
sqlStatementInput1 = "select * from " + database_name + "." + table_score + "_vars"

# Retrieve VARs and VALs data.
bbf1 = sqlContext.sql(sqlStatementInput1).groupBy("account_key").pivot("var").agg({"val": "sum"})

# Get the Discount for each Account_Key.
sqlStatementInput2 = "select account_key, val as discount from " + database_name + "." + table_score

# Retrieve Discount Data.
bbf2 = sqlContext.sql(sqlStatementInput2)   

# Join the two DataFrames and get them the correct data types.
bbf = bbf1.join(bbf2,["account_key"])

# Casting columns.
condition = lambda col: 'F_' in col
fcols = bbf.select(*filter(condition,bbf.columns)).columns
for colname in fcols:
    bbf = bbf.withColumn("temp", bbf[colname].cast("int")).drop(colname).withColumnRenamed("temp", colname)

condition = lambda col: 'R_' in col
rcols = bbf.select(*filter(condition,bbf.columns)).columns
for colname in rcols:
    bbf = bbf.withColumn("temp", bbf[colname].cast("int")).drop(colname).withColumnRenamed("temp", colname)
    
condition = lambda col: 'M_' in col
mcols = bbf.select(*filter(condition,bbf.columns)).columns
for colname in mcols:
    bbf = bbf.withColumn("temp", bbf[colname].cast("float")).drop(colname).withColumnRenamed("temp", colname)

dcols = ["discount", "AGE", "ANNIV", "ANNIV_DISC", "FEMALE", "INCOME", "KC_OPEN", "MALE", "MARRIED", "POC", "PREVIOUS_DISCOUNT"]
for colname in dcols:
    bbf = bbf.withColumn("temp", bbf[colname].cast("float")).drop(colname).withColumnRenamed("temp", colname)
########################################################################################
# Done Get SCORE DATA.   

# Save the {Spark}_driver into Hive.
if flag_score == 'true':
    name_table = 'ce_20180302_score_records_spark'
    df = bbf
else:
    name_table = 'ce_20180302_train_records_spark'
    
try:
    sqlContext.sql("use " + database_name)
    df.write.format("orc").saveAsTable(name_table, mode="overwrite")
except Exception, err:
    rt.print_exc()
    raise Exception("writing to Hive table fails")
    
print("Done Successfully.")

