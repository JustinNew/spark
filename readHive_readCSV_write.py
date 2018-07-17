#########################################################################################################################################
#
#  Author: Justin Tian
#  
#  Discount Tree Rank Correlation Between SAS and Spark 
#
#########################################################################################################################################

from __future__ import print_function

from pyspark.sql import SQLContext, HiveContext, SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.sql.window import Window
from pyspark.ml import Pipeline
from pyspark.mllib.stat import Statistics

import sys, re, getopt

# Copy program from local to remote server:
# scp Willis_write_finan_po.py 10.206.52.22:/home/tkg8w58/projects/DiscountTree
# hadoop fs -ls gs://kohls-ddh-lle-spark-root/DiscountTree

# Run the program as:
command_line = "nohup spark-submit --queue long_running --executor-memory 40G --num-executors 6 --jars /opt/hadoop/hadoop-2.8.3/share/hadoop/common/lib/gcs-connector-1.6.5-hadoop2-shaded.jar Willis_write_finan_po.py >& read_write.out &"

if __name__ == '__main__':

    #########################################################################
    # Create a spark Session.

    spark = SparkSession.builder.appName("Discount Tree").enableHiveSupport().getOrCreate()

    sc = spark.sparkContext
    sc.setLogLevel('FATAL')

    sqlContext = SQLContext(sc)

    #########################################################################
    #
    # Query to get data
    #
    #########################################################################

    # Get predictions for Spark
    sqlStatementInput1 = "select account_key, prediction, discount from db_spark_discount.ce_20180510_score_driver_spark where discount = 30"

    # Retrieve VARs and VALs data.
    df1 = sqlContext.sql(sqlStatementInput1)
    
    # Get predictions for SAS.
    sqlStatementInput2 = "select account_key, score, discount from db_sas_discount.ce_20180510_score_driver where discount = 30"

    # Retrieve VARs and VALs data.
    df2 = sqlContext.sql(sqlStatementInput2)
    
    # Join the Predictions for Discount = 15 between SAS and Spark.
    benchmark = df1.join(df2, ["account_key", "discount"])
    
    # Select example rows to display.
    print("Top 100 of Predictions.")
    benchmark.show(100)
    
    schema = StructType([
        StructField("NDX", IntegerType(), True),
        StructField("MKTG_CAL_DTE", StringType(), True),
        StructField("LY_DTE", StringType(), True),
        StructField("PO_DTE", StringType(), True)
        ])
    
    df = spark.read.schema(schema).option("header", "true").option("mode", "DROPMALFORMED").csv("gs://kohls-ddh-lle-spark-root/discount_optimization/finan_po.csv")
    
    # Save the {Spark}_driver into Hive.
    try:
        name_table = "finan_po_willis"
        sqlContext.sql("use db_stage")
        df.write.format("orc").saveAsTable(name_table, mode="overwrite")
    except Exception, err:
        raise Exception("writing to Hive table fails")

