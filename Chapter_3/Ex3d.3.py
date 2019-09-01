# Lasso regression
import pyspark
from pyspark.sql.functions import round
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, Bucketizer, OneHotEncoderEstimator, VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
import pandas as pd

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

# Create an indexer for carrier categorical feature
indexer = StringIndexer(inputCol="carrier", outputCol='carrier_idx')

# Indexer identifies categories in the data
indexer_model = indexer.fit(flights)

# Indexer creates a new column with numeric index values
flights_indexed = indexer_model.transform(flights)

# Repeat the process for the org categorical feature
flites = StringIndexer(inputCol="org", outputCol='org_idx').fit(flights_indexed).transform(flights_indexed)

# Create an instance of the one hot encoder
onehot = OneHotEncoderEstimator(inputCols=["org_idx"], outputCols=["org_dummy"])

# Apply the one hot encoder to the flights data
onehot = onehot.fit(flites)
flites = onehot.transform(flites)

pd.set_option('display.max_columns', None) # all cols
pd.set_option('display.width', 199)
pd.set_option('display.max_colwidth', 199)

# Create buckets at 3 hour intervals through the day
buckets = Bucketizer(splits=[0, 3, 6, 9, 12, 15, 18, 21, 24],
                     inputCol="depart", outputCol="depart_bucket")

# Bucket the departure times
bucketed = buckets.transform(flites)
#bucketed.select("depart", "depart_bucket").show(5)

# Create a one-hot encoder for departure
onehot = OneHotEncoderEstimator(inputCols=["depart_bucket"], outputCols=["depart_dummy"])

# One-hot encode the bucketed departure times
flites = onehot.fit(bucketed).transform(bucketed)

# Create a one-hot encoder for month
onehot = OneHotEncoderEstimator(inputCols=["mon"], outputCols=["mon_dummy"])

# One-hot encode the bucketed departure times
flites = onehot.fit(bucketed).transform(flites)

# Create a one-hot encoder for day of week
onehot = OneHotEncoderEstimator(inputCols=["dow"], outputCols=["dow_dummy"])

# One-hot encode the bucketed departure times
flites = onehot.fit(bucketed).transform(flites)
#flights_onehot.select("depart", "depart_bucket", "depart_dummy").show(5)

print("Sample model input")
print(flites.toPandas().sample(12))

# Create an assembler object
assembler = VectorAssembler(inputCols=["km", "org_dummy", "depart_dummy", "dow_dummy", "mon_dummy"],
                            outputCol='features')

# Consolidate predictor columns
flights_assembled = assembler.transform(flites)

# Check the resulting column
flites = flights_assembled.select('features', 'duration')

# Split the data into training and testing sets
flights_train, flights_test = flites.randomSplit([0.8, 0.2], seed=23)
#print(flights_train.toPandas().shape, flights_test.toPandas().shape)

# Create a lasso regression object and train on training data
lasso = LinearRegression(labelCol="duration", elasticNetParam=1, regParam=1).fit(flights_train)

# Create predictions for the testing data and take a look at the predictions
predictions = lasso.transform(flights_test)
#predictions.select('duration', 'prediction').show(truncate=False)
print("\nLasso Regression")
print(predictions.toPandas().sample(12))

# Calculate the RMSE
print("\nRMSE", RegressionEvaluator(labelCol="duration").evaluate(predictions))

# Print the coefficients and intercept for linear regression
print("\nCoefficients: %s" % str(lasso.coefficients))
print("Intercept: %s" % str(lasso.intercept))

# Summarize the model over the training set and print out some metrics
trainingSummary = lasso.summary
#print("numIterations: %d" % trainingSummary.totalIterations)
#print("objectiveHistory: %s\n" % str(trainingSummary.objectiveHistory))
#trainingSummary.residuals.show(8)
print("\nRMSE: %f" % trainingSummary.rootMeanSquaredError)
print("r2: %f" % trainingSummary.r2)

# Number of zero coefficients
zero_coeff = sum([beta == 0 for beta in lasso.coefficients])
print("\nNumber of ceofficients equal to 0:", zero_coeff)

# Average speed in km per hour
avg_speed = lasso.intercept / lasso.coefficients[0]
print("\nAverage speed in km/h", avg_speed)

spark.stop()
