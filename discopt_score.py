#########################################################################################################################################
#
#  Author: Justin Tian
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
from pyspark.ml.regression import DecisionTreeRegressor, DecisionTreeRegressionModel
from pyspark.ml.regression import RandomForestRegressor, RandomForestRegressionModel
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.stat import Statistics

import sys, re

# Copy program from local to remote server:
# scp discopt_score.py gcp-ddh-dev.tst.kohls.com:/home/tkg8w58/projects/DiscountTree
# scp discopt_score.py 10.206.52.22:/home/tkg8w58/projects/DiscountTree
# hadoop fs -ls gs://kohls-ddh-lle-spark-root/discount_optimization/discopt_model_15

# Run the program as:
command_line = "nohup spark-submit --queue long_running --executor-memory 40G --num-executors 6 --jars /opt/hadoop/hadoop-2.8.2/share/hadoop/common/lib/gcs-connector-1.6.0-hadoop2-shaded.jar discopt_score.py 'ce_20180302' 'db_sas_discount' 'ce_20180302_score' 'decisiontree' '15,20,30' 'gs://kohls-ddh-lle-spark-root/discount_optimization/' >& discopt_score.out &"

# Command line parameters.
if len(sys.argv) == 7:

    model_name     = sys.argv[1]
    database_name  = sys.argv[2]
    table_score    = sys.argv[3]
    flag_model     = sys.argv[4]
    discounts     = sys.argv[5]
    save_directory = sys.argv[6]

else:

    print("Using the default parameters." + command_line)
    model_name     = 'ce_20180302'
    database_name  = 'db_sas_discount'  
    table_score    = 'ce_20180302_score'
    flag_model     = 'decisiontree' # Or, 'randomforest'
    discounts     = '15,20,30'
    save_directory = 'gs://kohls-ddh-lle-spark-root/discount_optimization/'

# Change the discount levels into integers.
discounts = discounts.split(",")
discounts = [int(i) for i in discounts]

#########################################################################
# Create a spark Session.

spark = SparkSession.builder.appName("Discount Optimization").enableHiveSupport().getOrCreate()

sc = spark.sparkContext
sc.setLogLevel('FATAL')

sqlContext = SQLContext(sc)

#########################################################################
#
# Query to retrieve the customer data incl. account_key, var, val and trgt.
#
#########################################################################

print("#########################################################################")
print("Running Discount Optimization Spark Code To Score.")
print("The model for sales prediction is " + flag_model + ".")
print("The output is the sales predictions for all account_keys in SCORE dataset.") 
print("#########################################################################")

# There are three kinds of records: Discount = 15/20/30. 
# Get the VARs and associated VALs for each of the Account_Key.
########################################################################################
# Get SCORE DATA.
# Get the VARs and associated VALs for each of the Account_Key.
sqlStatementInput1 = "select * from " + database_name + "." + table_score + "_vars"

# Retrieve VARs and VALs data.
bbf = sqlContext.sql(sqlStatementInput1).groupBy("account_key").pivot("var").agg({"val": "sum"})

# Casting columns.
condition = lambda col: 'F_' in col
fcols = bbf.select(*filter(condition,bbf.columns)).columns
for colname in fcols:
    bbf = bbf.withColumn(colname, bbf[colname].cast("int"))

condition = lambda col: 'R_' in col
rcols = bbf.select(*filter(condition,bbf.columns)).columns
for colname in rcols:
    bbf = bbf.withColumn(colname, bbf[colname].cast("int"))
    
condition = lambda col: 'M_' in col
mcols = bbf.select(*filter(condition,bbf.columns)).columns
for colname in mcols:
    bbf = bbf.withColumn(colname, bbf[colname].cast("float"))

dcols = ["AGE", "ANNIV", "ANNIV_DISC", "FEMALE", "INCOME", "KC_OPEN", "MALE", "MARRIED", "POC", "PREVIOUS_DISCOUNT"]
for colname in dcols:
    bbf = bbf.withColumn(colname, bbf[colname].cast("int"))
########################################################################################
# Done Get SCORE DATA.   

# Using VectorAssembler to create "features" for pyspark.ml.regression.DecisionTreeRegressor.
assPath = save_directory + "discopt_model_assembler"
assembler = VectorAssembler.load(assPath)
bbf = assembler.transform(bbf)
    
# Train the model on all the different discount levels.
predictions = {}
nums_p = {}
for i in discounts: 
    
    # Load model and SCORE on records.
    modelPath = save_directory + "discopt_model_" + model_name + "_" + str(i)
    
    # Decide which model to use.
    if flag_model == 'randomforest':
        model = RandomForestRegressionModel.load(modelPath)
    else:
        # Use "minInstancesPerNode=2000" for full samples.
        model = DecisionTreeRegressionModel.load(modelPath)
    
    predictions[i] = model.transform(bbf)

# Add a new column as DISCOUNT label column.
cols = ['account_key', 'prediction', 'discount']
for i in discounts:
    predictions[i] = predictions[i].withColumn("discount", lit(i)).select(cols)
    nums_p[i] = predictions[i].count()
    print("Number of records in predictions " + str(i) + ' is: ' + str(nums_p[i]))

# Join all the predictions for different discount tree.
if len(discounts) == 1:
    all_predictions = predictions[discounts[0]]
elif len(discounts) == 2:
    all_predictions = predictions[discounts[0]].union(predictions[discounts[1]])
else:
    all_predictions = predictions[discounts[0]].union(predictions[discounts[1]])
    for i in range(2, len(discounts)):
        all_predictions = all_predictions.union(predictions[discounts[i]])

# Select example rows to display.
print("Top 100 of Predictions.")
all_predictions.select("account_key", "prediction", "discount").show(100)

print("The average of Spark Predictions for different discount levels.") 
for i in discounts:
    print('For discount level ' + str(i) + ':')
    predictions[i].select(mean("prediction")).show()

# Save the {Spark}_driver into Hive.
try:
    name_table = table_score + "_driver_spark"
    sqlContext.sql("use " + database_name)
    all_predictions.write.format("orc").saveAsTable(name_table, mode="overwrite")
except Exception, err:
    rt.print_exc()
    raise Exception("writing to Hive table fails")

