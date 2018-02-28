
# Spark and PySpark

combineByKey() explained very well here.

http://abshinn.github.io/python/apache-spark/2014/10/11/using-combinebykey-in-apache-spark/

https://github.com/apache/spark/tree/master/examples/src/main/python

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
