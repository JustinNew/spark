from pyspark.ml.classification import LogisticRegression
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml import Pipeline


# Create a SparkSession.
spark = SparkSession.builder.appName("Banknote_Logistic").getOrCreate()

# Load data and transfer into DataFrame.    
lines = spark.sparkContext.textFile("/Users/tkg8w58/Documents/Work/Projects/GCP_Spark/banknote/data_banknote_authentication.txt")

# Split line and create columns/variables.
temp_var = lines.map(lambda x: x.split(","))

# Convert to DataFrame.
df = temp_var.toDF(['var1','var2','var3','var4','label'])

# Select variable columns and casting String to Numeric.
df_num = df.select(df.var1.cast("float"), df.var2.cast("float"), df.var3.cast("float"), df.var4.cast("float"),df.label.cast("int"))

# Using VectorAssembler to create "features" for ml.LogisiticRegression.
label_stringIdx = StringIndexer(inputCol = "label", outputCol = "labelNew")
assembler = VectorAssembler(inputCols=[x for x in df_num.columns],outputCol='features')

# Using pipeline to combine multiple steps. 
pipeline = Pipeline(stages=[label_stringIdx, assembler])
pipelineModel = pipeline.fit(df_num)
df_new = pipelineModel.transform(df_num)

# Select the useful columns.
df_new = df_new.select(["labelNew", "features"])

(trainingData, testData) = df_new.randomSplit([0.7, 0.3], seed = 100)

# Create initial LogisticRegression model
# "label" is one column.
# "features" is one column, each element in the column is a vector.
lr = LogisticRegression(labelCol="labelNew", featuresCol="features", maxIter=10)

# Train model with Training Data
lrModel = lr.fit(trainingData)

# Make predictions on test data using the transform() method.
# LogisticRegression.transform() will only use the 'features' column.
predictions = lrModel.transform(testData)

# Select example rows to display.
predictions.select(["prediction", "labelNew", "features"]).show(5)
    
f = open('/Users/tkg8w58/Documents/Work/Projects/GCP_Spark/banknote/output.txt','w')
# Select (prediction, true label) and compute test error
# metricName: areaUnderPR or areaUnderROC
evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction", labelCol="labelNew", metricName="areaUnderROC")
areaUnderROC = evaluator.evaluate(predictions)

# Or accuracy.
# evaluator = MulticlassClassificationEvaluator(labelCol="labelNew", predictionCol="prediction", metricName="accuracy")
# accuracy = evaluator.evaluate(predictions)

f.write("Test Error = %g " % (1.0 - areaUnderPR))
f.close()


