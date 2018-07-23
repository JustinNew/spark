#########################################################################################################################################
#
#  Author: Justin Tian
#  
#  Discount Tree Optimization Spark Code -- Optimization
#  This code takes in account_key, prediction and discount from table_driver as input.
#  It compares with table_base to optimize the discount in table_driver 
#  so that the number of discounts match with table_base for each type.
#
#########################################################################################################################################

from __future__ import print_function

from pyspark.sql import SQLContext, HiveContext, SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.sql.window import Window
# from pyspark.sql.functions import desc, row_number

import sys, re, getopt

# Copy program from local to remote server:
# scp discopt_optimization.py gcp-ddh-dev.tst.kohls.com:/home/tkg8w58/projects/DiscountTree
# scp discopt_optimization.py 10.206.52.22:/home/tkg8w58/projects/DiscountTree
# hadoop fs -ls gs://kohls-ddh-lle-spark-root/discount_optimization/discopt_model_15
# nohup spark-submit --queue adhoc --executor-memory 40G --num-executors 6 --jars /opt/hadoop/hadoop-2.8.3/share/hadoop/common/lib/gcs-connector-1.6.5-hadoop2-shaded.jar discopt_optimization.py --title=ce_20180510 --database=db_spark_discount --basetable=ce_20180510_score --drivertable=ce_20180510_score_driver_spark --discounts=15,20,30  >& discopt_optimization.out &

# Run the program as:
command_line = "nohup spark-submit --queue long_running --executor-memory 40G --num-executors 6 --jars /opt/hadoop/hadoop-2.8.3/share/hadoop/common/lib/gcs-connector-1.6.5-hadoop2-shaded.jar discopt_optimization.py --title=ce_20180510 --database=db_spark_discount --basetable=ce_20180510_score --drivertable=ce_20180510_score_driver_spark --discounts=15,20,30  >& discopt_optimization.out &"

if __name__ == '__main__':
    
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'x', ['title=', 'database=', 'basetable=', 'drivertable=', 'discounts='])
    except getopt.GetoptError as err:
        # print help information and exit:
        print(err) # will print something like "option -a not recognized"
        usage()
        sys.exit(2)
    
    # Command line parameters.
    if len(opts) == 5:

        for l, v in opts:
            if l == '--title':
                model_name = v
            elif l == '--database':
                database_name = v
            elif l == '--basetable':
                table_base = v
            elif l == '--drivertable':
                table_driver = v
            elif l == '--discounts':
                discounts = v

    else:

        print("Using the default parameters." + command_line)
        model_name     = 'ce_20180510'
        database_name  = 'db_spark_discount'  
        table_base    = 'ce_20180510_score'
        table_driver     = 'ce_20180510_score_driver_spark'
        discounts     = '15,20,30'

    # Change the discount levels into integers.
    discounts = discounts.split(",")
    discounts = [int(i) for i in discounts]
    discounts.sort()

    #########################################################################
    # Create a spark Session.

    spark = SparkSession.builder.appName("Discount Optimization").enableHiveSupport().getOrCreate()

    sc = spark.sparkContext
    sc.setLogLevel('FATAL')

    sqlContext = SQLContext(sc)

    #########################################################################
    #
    # Query to retrieve the driver records to get account_key, discount, prediction.
    #
    #########################################################################

    print("#########################################################################")
    print("Running Discount Optimization Spark Code For Optimization.")
    print("The output is the final discount for each of the account_key.") 
    print("#########################################################################")

    # There are three kinds of records: Discount = 15/20/30. 
    ########################################################################################
    # Get account_key, discount from BASE.
    
    sqlStatementInput1 = "select account_key, discount from " + database_name + "." + table_base

    # Retrieve account_key, discount.
    bbf = sqlContext.sql(sqlStatementInput1)
    
     # Get account_key, discount and predictions/Sales from DRIVER.
    sqlStatementInput2 = "select account_key, discount, prediction from " + database_name + "." + table_driver

    # Retrieve account_key, discount and predictions/Sales.
    driver = sqlContext.sql(sqlStatementInput2)

    ########################################################################################
    # Done Get DATA.

    # Get the number of discounts in BASE table.
    print('In BASE:')
    num_discount = {}
    for i in discounts:
        num_discount[i] = bbf.filter(bbf.discount == i).count()
        print('Number of discount ' + str(i) + ' is ' + str(num_discount[i]))
    
    # Partition and Order By Prediction    
    driver = driver.withColumn('rowNumber', row_number().over(Window.partitionBy("account_key").orderBy(col("prediction").desc())))
    
    driver.show(30)
    
    # Count number for each of the discount before optimization.
    print('Before optimization, in DRIVER:')
    now_discount = {}
    for i in discounts:
        now_discount[i] = driver.filter((driver.discount == i) & (driver.rowNumber == 1)).count()
        print('Number of discount ' + str(i) + ' is ' + str(now_discount[i]))
    
    # Do a transformation of the DataFrame to have three DataFrames.
    # 1. account_key, discount_1, prediction_1
    # 2. account_key, discount_2, prediction_2
    # 3. account_key, discount_3, prediction_3
    df_discount = {}
    for i in range(1, len(discounts) + 1):    
        df_discount[i] = driver.filter(driver.rowNumber == i)
        t_d = 'discount_' + str(i)
        t_p = 'prediction_' + str(i)
        df_discount[i] = df_discount[i].withColumn(t_d, df_discount[i].discount).drop('discount')
        df_discount[i] = df_discount[i].withColumn(t_p, df_discount[i].prediction).drop('prediction')
        df_discount[i] = df_discount[i].drop('rowNumber')
    
    # Join all three separate DataFrames.
    # account_key, Discount, discount_1, prediction_1, discount_2, prediction_2, discount_3, prediction_3
    for i in range(len(discounts)):
        if i == 0:
            df_all = df_discount[i + 1]
        else:
            df_all = df_all.join(df_discount[i + 1], ['account_key'])
    df_all = df_all.withColumn('Discount', df_all.discount_1)
            
    df_all.show(10)
    
    # First Round Optimization.
    # Get the difference between prediction_1 and prediction_2 as diff_pred1_pred2
    # Order Discount desc, diff_pred1_pred2
    
    # From 30 to 20 to 15, get the first discount with more numbers in driver. 
    for i in discounts[::-1]:
        if now_discount[i] > num_discount[i]:
            cur_discount = i
            num_need = num_discount[i]
            break
            
    # Change the excess discount to second tier discount.   
    def func(pred1, pred2, Discount):
        if Discount == cur_discount:
            return pred1 - pred2
        else:
            return 0

    func_udf = udf(func, DoubleType())
    df_all = df_all.withColumn('diff_pred1_pred2',func_udf(df_all['prediction_1'], df_all['prediction_2'], df_all['Discount']))
    df_all = df_all.withColumn('index', row_number().over(Window.partitionBy("Discount").orderBy(col("diff_pred1_pred2").desc())))
    
    def change(Discount, discount2, index):
        if Discount != cur_discount:
            return Discount
        elif index > num_need:
            return discount2
        else:
            return Discount

    change_udf = udf(change, IntegerType())
    df_all = df_all.withColumn('Discount', change_udf(df_all['Discount'], df_all['discount_2'], df_all['index']))
    
    print('After first round optimization, in DRIVER:')
    now_discount = {}
    for i in discounts:
        now_discount[i] = df_all.filter(df_all.Discount == i).count()
        print('Number of discount ' + str(i) + ' is ' + str(now_discount[i]))
        
    # Second Round Optimization.
    # From 30 to 20 to 15, get the first discount with more numbers in driver. 
    for i in discounts[::-1]:
        if now_discount[i] > num_discount[i]:
            discount_toDecreaseNum = i
            num_toDecrease = now_discount[i] - num_discount[i]
            break
            
    for i in discounts[::-1]:
        if now_discount[i] < num_discount[i]:
            discount_toIncreaseNum = i
            break
            
    def diff(pred1, pred2, pred3, dscnt1, dscnt2, dscnt3, Discount):
        # Only deal with records not considered or optimized in first round.
        # Get the difference between the left two discount.
        if Discount != cur_discount:
            # Get the sales difference between higher tier discount and lower tier discount. 
            if dscnt1 == cur_discount:
                return pred2 - pred3
            elif dscnt2 == cur_discount:
                return pred1 - pred3
            else:
                return pred1 - pred2
        else:
            return 0

    diff_udf = udf(diff, DoubleType())
    df_all = df_all.withColumn('diff_12_23',diff_udf(df_all['prediction_1'], df_all['prediction_2'], df_all['prediction_3'], df_all['discount_1'], df_all['discount_2'], df_all['discount_3'],df_all['Discount']))
    # Just order by sales difference.
    # Order from smallest difference to largest difference, and change the smallest difference to the other discount. 
    df_all = df_all.withColumn('index', row_number().over(Window.partitionBy("Discount").orderBy(col("diff_12_23")))) 
    
    # Flip the excess discount_toDecreaseNum to discount_toIncreaseNum.
    # Flip the first num_toDecrease account_key for whom their sales difference between discount_toDecreaseNum and discount_toIncreaseNum are smallest.
    def change2(Discount, index):
        if Discount == cur_discount:
            return Discount
        elif Discount == discount_toDecreaseNum and index <= num_toDecrease:
            return discount_toIncreaseNum
        else:
            return Discount

    change2_udf = udf(change2, IntegerType())
    df_all = df_all.withColumn('Discount', change2_udf(df_all['Discount'], df_all['index']))
    
    print('After second round optimization, in DRIVER:')
    now_discount = {}
    for i in discounts:
        now_discount[i] = df_all.filter(df_all.Discount == i).count()
        print('Number of discount ' + str(i) + ' is ' + str(now_discount[i]))
        
    # Get the final discount and sales for each account_key.
    def finalSales(pred1, pred2, pred3, dscnt1, dscnt2, dscnt3, Discount):

        if Discount == dscnt1:
            return pred1
        elif Discount == dscnt2:
            return pred2
        else:
            return pred3

    finalSales_udf = udf(finalSales, DoubleType())
    df_all = df_all.withColumn('Sales',finalSales_udf(df_all['prediction_1'], df_all['prediction_2'], df_all['prediction_3'], df_all['discount_1'], df_all['discount_2'], df_all['discount_3'],df_all['Discount']))
    
    # Get final table.
    df_final = df_all.select(['account_key', 'Discount', 'Sales'])
    
    # Save the {Spark}_final into Hive.
    try:
        name_table = table_base + "_final_spark"
        sqlContext.sql("use " + database_name)
        df_final.write.format("orc").saveAsTable(name_table, mode="overwrite")
    except Exception, err:
        rt.print_exc()
        raise Exception("writing to Hive table fails")

