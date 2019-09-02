# derived from EX3b.1
import pyspark
from pyspark.sql.functions import round
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, OneHotEncoderEstimator, VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator

REGRESSION_STAGE = 3

# Creating a SparkSession
spark = SparkSession.builder \
                    .master('local[*]') \
                    .appName('first_spark_application') \
                    .getOrCreate()

flights = spark.read.csv('/Users/wel51x/Box Sync/MyBox/Code/DataCamp/data/flights.csv',
#flights = spark.read.csv('/Users/wel51x/Box Sync/MyBox/Code/DataCamp/data/flights_truncated.csv',
                         sep=',',
                         header=True,
                         inferSchema=True,
                         nullValue='NA')

# Get number of records
print("The data contain %d records." % flights.count())

# Remove the 'flight' column
flights = flights.drop('flight')

# Convert 'mile' to 'km' and drop 'mile' column
flights = flights.withColumn('km', round(flights.mile * 1.60934, 0)) \
                 .drop('mile')

# Remove records with missing values in any column and get the number of remaining rows
flights = flights.dropna()
print("The data contains %d records after dropping records with na values." % flights.count())

flites = flights.select('km', 'org', 'dow', 'duration')

print("Sample model input")
print(flites.toPandas().sample(12))

# Create an indexer for the org categorical feature
indexer = StringIndexer(inputCol="org", outputCol='org_idx')

# Create an OneHot Encoder for the org & dow categorical features
onehot = OneHotEncoderEstimator(inputCols=['org_idx', 'dow'],
                                outputCols=['org_dummy', 'dow_dummy'])

# Create 'features' vector: 'km', 'org_dummy', 'dow_dummy'
assembler = VectorAssembler(inputCols=['km', 'org_dummy', 'dow_dummy'], outputCol='features')

# Split the data into training and testing sets
flights_train, flights_test = flites.randomSplit([0.8, 0.2], seed=23)

# Create a regression object and train on training data
regression = LinearRegression(labelCol="duration")

# Combine steps into a pipeline
pipeline = Pipeline(stages=[indexer, onehot, assembler, regression])

spark.stop()
