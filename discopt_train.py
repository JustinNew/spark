#########################################################################################################################################
#
#  Author: Justin Tian
#  
#  Discount Tree Optimization Spark Code To Build Model(s)
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

# Copy program from local to remote server:
# scp discopt_train.py gcp-ddh-dev.tst.kohls.com:/home/tkg8w58/projects/DiscountTree
# scp discopt_train.py 10.206.52.22:/home/tkg8w58/projects/DiscountTree
# Check written files:
# hadoop fs -ls
# hadoop fs -ls gs://kohls-ddh-lle-spark-root/discount_optimization/discopt_model_15

# Run the program as:
command_line = "nohup spark-submit --queue long_running --executor-memory 40G --num-executors 6 discopt_train.py 'ce_20180302' 'db_sas_discount' 'data' 'decisiontree' '15,20,30' 'gs://kohls-ddh-lle-spark-root/discount_optimization/' >& discopt_train.out &"

# Command line parameters.
if len(sys.argv) == 7:

    model_name     = sys.argv[1]
    database_name  = sys.argv[2]
    table_train    = sys.argv[3]
    flag_model     = sys.argv[4]
    discounts      = sys.argv[5]
    save_directory = sys.argv[6]

else:

    print("Using the default parameters." + command_line)
    model_name     = 'ce_20180302'
    database_name  = 'db_sas_discount'
    table_train    = 'data'   
    flag_model     = 'decisiontree' # Or, 'randomforest'
    discounts      = '15,20,30'
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
# Query to retrieve the customer data incl. sales, demos and target data
#
#########################################################################

print("#########################################################################")
print("Running Discount Optimization Model.")
print("The model(s) for sales prediction is " + flag_model + ".")
print("The outputs are models for discounts in " + str(discounts) + " each.") 
print("#########################################################################")

# There are three kinds of records: Discount = 15/20/30. 

########################################################################################
# Gets inputs as key value pairs (var & val) for each customer
sqlStatementInputs1 = "select account_key, var, val from " + database_name + "." + table_train

# transposes th input data data to get one record per customer  
df1 = sqlContext.sql(sqlStatementInputs1).groupBy("account_key").pivot("var").agg({"val": "sum"})

# Gets the Discount and Target for each customer
sqlStatementInputs2 = "select distinct account_key, discount, trgt from " + database_name + "." + table_train

# Retrieve Discount and TRGT Data.
df2 = sqlContext.sql(sqlStatementInputs2)     

# Join inputs with target and discount
df = df1.join(df2,["account_key"])

# Casting "F_" input columns.
condition = lambda col: 'F_' in col
fcols = df.select(*filter(condition,df.columns)).columns
for colname in fcols:
    df = df.withColumn(colname, df[colname].cast("int"))

# Casting "R_" input columns.
condition = lambda col: 'R_' in col
rcols = df.select(*filter(condition,df.columns)).columns
for colname in rcols:
    df = df.withColumn(colname, df[colname].cast("int"))
    
# Casting "M_" input columns.
condition = lambda col: 'M_' in col
mcols = df.select(*filter(condition,df.columns)).columns
for colname in mcols:
    df = df.withColumn(colname, df[colname].cast("float"))
    
# Casting remaining input columns.
dcols = ["AGE", "ANNIV", "ANNIV_DISC", "FEMALE", "INCOME", "KC_OPEN", "MALE", "MARRIED", "POC", "PREVIOUS_DISCOUNT"]
for colname in dcols:
    df = df.withColumn(colname, df[colname].cast("int"))
    
# casts the target column
df = df.withColumn("trgt", df.trgt.cast("float"))

# casts the discount column
df = df.withColumn("discount", df.discount.cast("int"))

num_obs = df.count()

# Number of observations
print("Number of observations read: " + str(num_obs))

# Summary about the DataFrame.
df.describe().show()

# Count the NULL values in each of the columns.
print('Results about missing value counts.')
df.select([count(when(isnan(c), c)).alias(c) for c in df.columns]).show()

# Replaces missing values by 0
df = df.na.fill(0)

########################################################################################

# Using VectorAssembler to create "features"
assembler = VectorAssembler(inputCols=[x for x in fcols + rcols + mcols + dcols], outputCol='features')
df = assembler.transform(df)
assPath = save_directory + "discopt_model_assembler"
assembler.write().overwrite().save(assPath)

########################################################################################

# Train a tree based model.
if flag_model == 'randomforest':
    dt = RandomForestRegressor(featuresCol="features", labelCol="trgt", maxDepth=5, numTrees=50)
else:
    # Use "minInstancesPerNode=2000" for full samples.
    dt = DecisionTreeRegressor(featuresCol="features", labelCol="trgt", maxDepth=8, minInstancesPerNode=2000)

# Train the model on all discount levels.
for i in discounts: 
    
    # Get trainData for all discount levels.
    trainData = df.filter(df.discount == i)
    print('For discount level ' + str(i) + ' the trgt average:')
    trainData.select(mean("trgt")).show()

    # Train model on trainData and Score on testData.
    model = dt.fit(trainData)
    
    # Save the trained model.
    modelPath = save_directory + "discopt_model_" + model_name + "_" + str(i)
    model.write().overwrite().save(modelPath)
    #print(model)
    print('Written ' + modelPath + ' Successful.')

