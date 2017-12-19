# This uses the adult_data.txt from UCI Machine Learning Repository.

from pyspark.sql import SparkSession
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, FloatType
from pyspark.sql import Row
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

# Create SparkSession.
spark = SparkSession.builder.appName("Adult_DecisionTree").getOrCreate()

#################################################################################################################################
# Read in Data and Create DataFrame.
#################################################################################################################################
# Option 1: read in txt as RDD and process. Gives error for this data.
if (1 == 2) {
    train_lines = spark.sparkContext.textFile("/Users/tkg8w58/Documents/Work/Projects/GCP_Spark/adult/adult_data.txt")
    test_lines = spark.sparkContext.textFile("/Users/tkg8w58/Documents/Work/Projects/GCP_Spark/adult/adult_test.txt")

    # Create variables.
    train_vars = train_lines.map(lambda x: x.split(', '))
    test_vars = test_lines.map(lambda x: x.split(', '))

    # Using Row to assign correct types.
    train_typed = train_vars.map(lambda x: Row(age = float(x[0]), workclass = str(x[1]), fnlwgt = float(x[2]), education = str(x[3]), \
                                           education_num = float(x[4]), marital_status = str(x[5]), occupation = str(x[6]), \
                                           relationship = str(x[7]), race = str(x[8]), sex = str(x[9]), capital_gain = float(x[10]), \
                                           capital_loss = float(x[11]), hours_per_week = float(x[12]), native_country = str(x[13]), \
                                           income = str(x[14])))
                                           
    test_typed = train_vars.map(lambda x: Row(age = float(x[0]), workclass = str(x[1]), fnlwgt = float(x[2]), education = str(x[3]), \
                                           education_num = float(x[4]), marital_status = str(x[5]), occupation = str(x[6]), \
                                           relationship = str(x[7]), race = str(x[8]), sex = str(x[9]), capital_gain = float(x[10]), \
                                           capital_loss = float(x[11]), hours_per_week = float(x[12]), native_country = str(x[13]), \
                                           income = str(x[14])))                                              


    # Create DataFrame
    df_train = spark.createDataFrame(train_typed)    
    df_test = spark.createDataFrame(test_typed)  
}

#################################################################################################################################
# Option 2: Read in txt as RDD and Row. Gives error for this data.
if ( 1 == 2) {
    train_lines = spark.read.text("/Users/tkg8w58/Documents/Work/Projects/GCP_Spark/adult/adult_data.txt").rdd
    train_vars = train_lines.map(lambda x: x.value.split(', '))

    # Using Row to assign correct types.
    train_typed = train_vars.map(lambda x: Row(age = float(x[0]), workclass = str(x[1]), fnlwgt = float(x[2]), education = str(x[3]), \
                                           education_num = float(x[4]), marital_status = str(x[5]), occupation = str(x[6]), \
                                           relationship = str(x[7]), race = str(x[8]), sex = str(x[9]), capital_gain = float(x[10]), \
                                           capital_loss = float(x[11]), hours_per_week = float(x[12]), native_country = str(x[13]), \
                                           income = str(x[14])))
                                           
    # Create DataFrame
    df_train = spark.createDataFrame(train_typed)
}

#################################################################################################################################
# Option 3: Read in csv as DataFrame with types inferred or given by schema.

# Define schema and using spark.read.csv(..., schema=schema)
# schema = StructType([
#    StructField("age", FloatType(), True),
#    StructField("workclass", StringType(), True)])

# spark.read.csv() using inferSchema.
df = spark.read.csv("/Users/tkg8w58/Documents/Work/Projects/GCP_Spark/adult/adult_data.txt", header=False, sep=",", \
                    ignoreLeadingWhiteSpace=True, ignoreTrailingWhiteSpace=True, inferSchema=True)
                    
# Rename Column names.
df = df.withColumnRenamed('_c0', 'age') 
df = df.withColumnRenamed('_c1', 'workclass') 
df = df.withColumnRenamed('_c2', 'fnlwgt') 
df = df.withColumnRenamed('_c3', 'education') 
df = df.withColumnRenamed('_c4', 'education_num') 
df = df.withColumnRenamed('_c5', 'marital_status') 
df = df.withColumnRenamed('_c6', 'occupation') 
df = df.withColumnRenamed('_c7', 'relationship') 
df = df.withColumnRenamed('_c8', 'race') 
df = df.withColumnRenamed('_c9', 'sex') 
df = df.withColumnRenamed('_c10', 'capital_gain') 
df = df.withColumnRenamed('_c11', 'capital_loss') 
df = df.withColumnRenamed('_c12', 'hours_per_week')                                
df = df.withColumnRenamed('_c13', 'native_country')
df = df.withColumnRenamed('_c14', 'income')  

#################################################################################################################################                            
                            
categoricalColumns = ["workclass", "education", "marital_status", "occupation", "relationship", "race", "sex", "native_country"]

# Stages in our Pipeline
stages = [] 

# Feature engineerings.
for categoricalCol in categoricalColumns:
    
  # Category Indexing with StringIndexer
  stringIndexer = StringIndexer(inputCol=categoricalCol, outputCol=categoricalCol+"Index")
  
  # Use OneHotEncoder to convert categorical variables into binary SparseVectors
  encoder = OneHotEncoder(inputCol=categoricalCol+"Index", outputCol=categoricalCol+"classVec")
  
  # Add stages.  These are not run here, but will run all at once later on.
  stages += [stringIndexer, encoder]
  
# Convert label into label indices using the StringIndexer
label_stringIdx = StringIndexer(inputCol = "income", outputCol = "label")
stages += [label_stringIdx]

# Transform all features into a vector using VectorAssembler
numericCols = ["age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
assemblerInputs = [c+"classVec" for c in categoricalColumns] + numericCols
assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
stages += [assembler]

# Create a Pipeline.
pipeline = Pipeline(stages=stages)
# Run the feature transformations.
#  - fit() computes feature statistics as needed.
#  - transform() actually transforms the features.
pipelineModel = pipeline.fit(df)
df = pipelineModel.transform(df)

# Keep relevant columns
selectedcols = ["label", "features"]
df = df.select(selectedcols)

# Create initial Decision Tree Model
dt = DecisionTreeClassifier(labelCol="label", featuresCol="features", maxDepth=3)

# Train model with Training Data
dtModel = dt.fit(trainingData)

# Make predictions on test data using the Transformer.transform() method.
predictions = dtModel.transform(testData)

# Select example rows to display.
predictions.select(["prediction", "label"]).show(5)

# Evaluate model
evaluator = BinaryClassificationEvaluator()
evaluator.evaluate(predictions)