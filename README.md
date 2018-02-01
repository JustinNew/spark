
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
