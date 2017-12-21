# Import SparkSession
from pyspark.sql import SparkSession

# Create SparkSession
spark = SparkSession.builder.appName('test').getOrCreate()

# Read in csv file as DataFrame.
file = spark.read.csv('/Users/tjh/Documents/Projects/banknote/data_banknote_authentication.txt', header=False, sep=",", \
                      ignoreLeadingWhiteSpace=True, ignoreTrailingWhiteSpace=True, inferSchema=True)

# See the types of each column.
file.printSchema()

# Count the number of rows.
file.count()

# Get the column names.
file.columns

# Get the summary of the DataFrame.
file.describe().show()

# Select certain columns.
file.select('_c0','_c1').show()

# Count distinct values in one column.
file.select('_c4').distinct().count()

# Calculate pair wise frequency of categorical columns.
file.crosstab('_c0','_c4').show()

# Get the DataFrame which won’t have duplicate rows of given DataFrame.
file.select('_c0','_c4').dropDuplicates().show()

# Drop null values rows.
file.dropna().count()

# Fill null values.
file.fillna(-1).show(2)

# Filter the rows.
file.filter(file._c4 == 0).count()

# Aggregation and group.
file.groupby('_c4').agg({'_c0': 'mean', '_c1': 'sum'}).show()

# Sample data.
sample = file.sample(False, 0.2, 10)

# Order DataFrame on column.
file.orderBy(file._c4.desc(), file._c0.asc()).show(20)

# Add new column.
file.withColumn('new', file._c0*2).show(5)

# Drop column.
file.drop('_c0').columns 
