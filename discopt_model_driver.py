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

# Copy program from local to remote server:
# scp discopt_model_driver.py gcp-ddh-dev.tst.kohls.com:/home/tkg8w58/projects/DiscountTree
# scp discopt_model_driver.py 10.206.52.22:/home/tkg8w58/projects/DiscountTree

# Run the program as:
command_line = "nohup spark-submit --queue adhoc --executor-memory 40G --num-executors 6 discopt_model_driver.py 'DTO_Tree' 'db_sas_discount' 'ce_20180302_train' 'ce_20180302_score' 'decisiontree' '15,20,30' 'true' >& discopt_model_driver.out &"

# Command line parameters.
if len(sys.argv) == 8:

    model_name = sys.argv[1]
    database_name = sys.argv[2]
    table_train = sys.argv[3]
    table_score = sys.argv[4]
    flag_model = sys.argv[5]
    l_discount = sys.argv[6]
    flag_score = sys.argv[7]

else:

    print("Using the default parameters." + command_line)
    model_name = 'DTO_Tree'
    database_name = 'db_sas_discount'
    table_train = 'ce_20180302_train'   
    table_score = 'ce_20180302_score'
    flag_model = 'decisiontree' # Or, 'randomforest'
    l_discount = '15,20,30'
    flag_score = 'true'

# Change the discount levels into integers.
l_discount = re.findall('\d+', l_discount)
l_discount = [int(i) for i in l_discount]

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
print("Running Discount Optimization Spark Code.")
print("The model for sales prediction is " + flag_model + ".")
print("The output is the sales predictions for all account_keys in ", t_score, " dataset.") 
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

# Using VectorAssembler to create "features" for pyspark.ml.regression.DecisionTreeRegressor.
assembler = VectorAssembler(inputCols=[x for x in df.columns[3:-1]],outputCol='features')

# Train a DecisionTree/Random Forest model.
if flag_model == 'randomforest':
    dt = RandomForestRegressor(featuresCol="features", labelCol="trgt", maxDepth=5, numTrees=50)
else:
    # Use "minInstancesPerNode=2000" for full samples.
    dt = DecisionTreeRegressor(featuresCol="features", labelCol="trgt", maxDepth=8, minInstancesPerNode=2000)

# Using pipeline to combine multiple steps. 
pipeline = Pipeline(stages=[assembler, dt])

# Depending on TRAIN or SCORE, choose the testData.
if flag_score == 'true':
    testData = bbf
else:
    testData = df

# Train the model on all the different discount levels.
predictions = {}
nums_p = {}
for i in l_discount: 
    
    # Get trainData for all discount levels.
    trainData = df.filter(df.discount == i)

    # Train model on trainData and Score on testData.
    model = pipeline.fit(trainData)
    predictions[i] = model.transform(testData)

# Add a new column as DISCOUNT label column.
cols = ['account_key', 'prediction', 'dscnt']
for i in l_discount:
    predictions[i] = predictions[i].withColumn("dscnt", lit(i)).select(cols)
    nums_p[i] = predictions[i].count()
    print("Number of records in predictions " + str(i) + ' is: ' + str(nums_p[i]))

# Join all the predictions for different discount tree.
if len(l_discount) == 1:
    all_predictions = predictions[l_discount[0]]
elif len(l_discount) == 2:
    all_predictions = predictions[l_discount[0]].union(predictions[l_discount[1]])
else:
    all_predictions = predictions[l_discount[0]].union(predictions[l_discount[1]])
    for i in range(2, len(l_discount)):
        all_predictions = all_predictions.union(predictions[l_discount[i]])

# Select example rows to display.
print("Top 100 of Predictions.")
all_predictions.select("account_key", "prediction", "dscnt").show(100)

print("The average of Spark Predictions for different discount levels.") 
for i in l_discount:
    print('For discount level ' + str(i) + ':')
    predictions[i].select(mean("prediction")).show()

# Save the {Spark}_driver into Hive.
if flag_score == 'true':
    name_table = 'ce_20180302_score_driver_spark'
else:
    name_table = 'ce_20180302_train_driver_spark'
    
try:
    sqlContext.sql("use " + database_name)
    all_predictions.write.format("orc").saveAsTable(name_table, mode="overwrite")
except Exception, err:
    rt.print_exc()
    raise Exception("writing to Hive table fails")

