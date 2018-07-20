
Spark and PySpark
----------

combineByKey() explained very well here.

http://abshinn.github.io/python/apache-spark/2014/10/11/using-combinebykey-in-apache-spark/

https://github.com/apache/spark/tree/master/examples/src/main/python

### Cluster Usage

  - Check Jobs: yarn application -list
  - Kill Jobs: yarn application -kill jobid

### Copy File From Local To Hadoop

```bash
hadoop fs -put /path/in/linux /hdfs/path
```

### Python In PySpark

- In PySpark, we can write and have Python codes. But be careful, do not confuse Spark with Python codes. In other words, list operations, string operations, etc. can not be on the line of DataFrmame or SQL.

### Join Two DataFrames

```python
df = df1.join(df2,["account_key"])
```

Note: 
  
  - Use **['colname']** instead of **df1.col == df2.col**
  
### Rank Correlation

Get the rank correlation.

```python
firstRDD = benchmark.rdd.map(lambda x: x['prediction'])
secondRDD = benchmark.rdd.map(lambda x: x['score'])
corr = Statistics.corr(firstRDD, secondRDD, method="spearman")
print("The rank correlation between SAS and Spark Prediction = %g" % corr) 
```

### Rank () Over ()

Add a 'rank' column:

```python
from pyspark.sql.functions import *
from pyspark.sql.window import Window

ranked =  df.withColumn("rank", dense_rank().over(Window.partitionBy("A").orderBy(desc("C"))))
```

### Add New Columns

```python
from pyspark.sql.functions import lit
from pyspark.sql.functions import exp

df = sqlContext.createDataFrame(
    [(1, "a", 23.0), (3, "B", -23.0)], ("x1", "x2", "x3"))

##### Add a column with all '0'
df_with_x4 = df.withColumn("x4", lit(0))

ddff = ddff.withColumn('Discount, ddff['discount'].cast("int"))

##### Add new column with math and ordering
def func(pred1, pred2, Discount):
    if Discount == cur_discount:
        return pred1 - pred2
    else:
        return 0

func_udf = udf(func, IntegerType())
df_discount = df_discount.withColumn('dif_pred1_pred2',func_udf(df_discount['prediction_1'], df_discount['prediction_2'], df_discount['Discount']))
df_discount = df_discount.withColumn('index', orderBy(col('diff_pred1_pred2').desc()))
```

### Represent a column name with a variable

```python
from pyspark.sql.functions import col

i = 15
t_d = 'discount_' + str(i)
df_discount.withColumn(t_d, df_discount.discount)
```

### Write DataFrame into Hive

**all_predictions** is a DataFrame in PySpark.

```python
try:
    sqlContext.sql("use " + database_name)
    all_predictions.write.format("orc").saveAsTable('ce_20180302_train_driver_spark', mode="overwrite")
except Exception, err:
    rt.print_exc()
    raise Exception("writing to Hive table fails")
```

### Cast column to new type

```python
condition = lambda col: 'F_' in col
listcols = df.select(*filter(condition, df.columns)).columns

for colname in listcols:
    df = df.withColumn("temp", df[colname].cast("int")).drop(colname).withColumnRenamed("temp", colname)
	
df2 = df.withColumn("yearTmp", df.year.cast("int")).drop("year").withColumnRenamed("yearTmp", "year")
```

### filter by column value
```python
foo_df = df.filter( (df.foo==1) & (df.bar.isNull()) )
```

### Row_number

```python
from pyspark.sql.functions import desc, row_number 
  
driver = driver.withColumn('rowNumber', row_number().over(Window.partitionBy("account_key").orderBy(col("prediction").desc())))
```

### Save and Load ML Model

```python
from pyspark.ml.regression import DecisionTreeRegressor, RandomForestRegressor, DecisionTreeRegressionModel, RandomForestRegressionModel

dt = DecisionTreeRegressor(featuresCol="features", labelCol="trgt", maxDepth=8, minInstancesPerNode=2000)
model = dt.fit(trainData)
modelPath = "disopt_model_" + str(i)
model.write().overwrite().save(modelPath)

dtLoad = DecisionTreeRegressionModel.load(modelPath)
```

### PySpark Schema

```python
schema = StructType([
    StructField("A", IntegerType(), True),
    StructField("B", DoubleType(), True),
    StructField("C", StringType(), True)
])
```

Note:
  - **True** means the column allows null values.