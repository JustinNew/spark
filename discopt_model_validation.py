#########################################################################################################################################
#
#  Discount Tree Driver Comparison Code
#  This code takes in the $BASE_driver table from SAS model and Spark model and does comparison between the two tables.
#  It ranks the discounts for each of the account_key according to their expected sales from model generated in discopt_model_driver.py. 
#
#########################################################################################################################################

from __future__ import print_function

from pyspark.sql import SQLContext, HiveContext, SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.sql.window import Window
# from pyspark.ml.feature import VectorAssembler
# from pyspark.ml.regression import DecisionTreeRegressor, RandomForestRegressor
# from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
# from pyspark.mllib.stat import Statistics

import sys, re

# Copy program from local to remote server:
# scp discopt_model_validation.py gcp-ddh-dev.tst.kohls.com:/home/tkg8w58/projects/DiscountTree
# scp discopt_model_validation.py 10.206.52.22:/home/tkg8w58/projects/DiscountTree

# Run the program as:
command_line = "nohup spark-submit --queue adhoc --executor-memory 40G --num-executors 6 discopt_model_validation.py 'validation' 'db_sas_discount' 'ce_20180302_train' 'ce_20180302_score' '15,20,30' 'true' >& discopt_model_validation.out &"

# Command line parameters.
if len(sys.argv) == 7:

    model_name = sys.argv[1]
    database_name = sys.argv[2]
    table_train = sys.argv[3]
    table_score = sys.argv[4]
    l_discount = sys.argv[5]
    flag_score = sys.argv[6]

else:

    print("Using the default parameters." + command_line)
    model_name = 'validation'
    database_name = 'db_sas_discount'
    table_train = 'ce_20180302_train'   
    table_score = 'ce_20180302_score'
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
print("Running Discount Tree Driver Comparison Code.")
print("Compare the &BASE_dricver discount rank according to sales for " + t_score + " between SAS and Spark results.")
print("#########################################################################")

# There are about 25 M account_keys in the dataset. 
# There are three kinds of records: Discount = 15/20/30. 
########################################################################################
# Get SAS DRIVER DATA.
if flag_score == 'true':
    sqlStatementInputs1 = "select * from " + database_name + "." + table_score + "_driver"
else:
    sqlStatementInputs1 = "select * from " + database_name + "." + table_train + "_driver"

# Retrieve account_key, score and discount data.
df_sas = sqlContext.sql(sqlStatementInputs1)
df_sas = df_sas.select(df_sas.account_key, df_sas.score.cast("float"), df_sas.discount.cast("int"))
########################################################################################
# Done Get SAS DRIVER DATA.   
   
########################################################################################
# Get SPARK DRIVER DATA.          
if flag_score == 'true': 
    sqlStatementInputs2 = "select * from " + database_name + "." + table_score + "_driver_spark"
else:
    sqlStatementInputs2 = "select * from " + database_name + "." + table_train + "_driver_spark"
    
df_spark = sqlContext.sql(sqlStatementInputs2)
print(str(df_spark.columns))
df_spark = df_spark.select(df_spark.account_key, df_spark.prediction.cast("float"), df_spark.dscnt.cast("int"))
########################################################################################
# Done Get SPARK DRIVER DATA. 

# Number of observations in CE_20180302_train/score_driver
nums_df_spark = {}
nums_df_sas = {}
for i in l_discount:
    nums_df_spark[i] = df_spark.filter(df_spark.dscnt == i).count()
    nums_df_sas[i] = df_sas.filter(df_sas.discount == i).count()
    print("Number of observations read in ce_20180302_train/score_driver_spark with discount equals " + str(i) + " is " + str(nums_df_spark[i]))
    print("Number of observations read in ce_20180302_train/score_driver with discount equals " + str(i) + " is " + str(nums_df_sas[i]))
    print(" ")

print("\n\n\n")

# Get the first/second/... discount with according to predicted sales.
df_spark = df_spark.withColumn("rank", dense_rank().over(Window.partitionBy("account_key").orderBy(desc("prediction"))))
df_sas = df_sas.withColumn("rank", dense_rank().over(Window.partitionBy("account_key").orderBy(desc("score"))))

for i in range(len(l_discount)):
    
    t_spark = df_spark.filter(df_spark.rank == i + 1)
    t_sas = df_sas.filter(df_sas.rank == i + 1)

    nums_spark = t_spark.count()
    nums_sas = t_sas.count()
    print("Number of records in Spark rank " + str(i + 1) + " is " + str(nums_spark))
    print("Number of records in SAS rank " + str(i + 1) + " is " + str(nums_sas))
    print("\n")
    
    comparison = t_spark.join(t_sas, ["account_key"])
    nums_comparison = comparison.count()
    print("Number of records after join SAS and Spark rank " + str(i + 1) + " is " + str(nums_comparison))
    
    t_d = -1 * (i + 1)
    sas_t = comparison.filter(comparison.discount == l_discount[t_d]).count()
    spark_t = comparison.filter(comparison.dscnt == l_discount[t_d]).count()
    print("Number of Discount equals " + str(l_discount[t_d]) +  " in SAS Driver Rank " + str(i + 1) + " is " + str(sas_t))
    print("Number of Discount equals " + str(l_discount[t_d]) +  " in Spark Driver Rank " + str(i + 1) + " is " + str(spark_t))
    print("\n")
    
    tot_nums = comparison.count()
    print("Total records " + str(tot_nums))
    print("\n")
    
    match_nums = comparison.filter((comparison.dscnt == 15) & (comparison.discount == 15)).count()
    print("Records with spark 15% and sas 15% is " + str(match_nums))
    match_nums = comparison.filter((comparison.dscnt == 15) & (comparison.discount == 20)).count()
    print("Records with spark 15% and sas 20% is " + str(match_nums))
    match_nums = comparison.filter((comparison.dscnt == 15) & (comparison.discount == 30)).count()
    print("Records with spark 15% and sas 30% is " + str(match_nums))
    print("\n")
    
    match_nums = comparison.filter((comparison.dscnt == 20) & (comparison.discount == 15)).count()
    print("Records with spark 20% and sas 15% is " + str(match_nums))
    match_nums = comparison.filter((comparison.dscnt == 20) & (comparison.discount == 20)).count()
    print("Records with spark 20% and sas 20% is " + str(match_nums))
    match_nums = comparison.filter((comparison.dscnt == 20) & (comparison.discount == 30)).count()
    print("Records with spark 20% and sas 30% is " + str(match_nums))
    print("\n")
    
    match_nums = comparison.filter((comparison.dscnt == 30) & (comparison.discount == 15)).count()
    print("Records with spark 30% and sas 15% is " + str(match_nums))
    match_nums = comparison.filter((comparison.dscnt == 30) & (comparison.discount == 20)).count()
    print("Records with spark 30% and sas 20% is " + str(match_nums))
    match_nums = comparison.filter((comparison.dscnt == 30) & (comparison.discount == 30)).count()
    print("Records with spark 30% and sas 30% is " + str(match_nums))
    print("\n")
    
    evaluator = RegressionEvaluator(
        labelCol="score", predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(comparison)
    print("Root Mean Squared Error (RMSE) on SAS and Spark Prediction = %g" % rmse)
    
    print("\n\n\n")
    


