# derived from EX3b.1
import pyspark
from pyspark.sql.functions import round
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, OneHotEncoderEstimator, VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator

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
#flights_indexed = StringIndexer(inputCol="org", outputCol='org_idx').fit(flights).transform(flights)
indexer = StringIndexer(inputCol="org", outputCol='org_idx')
# Assign index values to strings
indexer = indexer.fit(flites)
# Create column with index values
flites = indexer.transform(flites)

# Check first five records
#flights_indexed.show(5)

onehot = OneHotEncoderEstimator(inputCols=['org_idx', 'dow'],
                                outputCols=['org_dummy', 'dow_dummy'])
flites = onehot.fit(flites).transform(flites)

# Create 'features' vector: 'weight_kg', 'cyl', 'type_dummy'
assembler = VectorAssembler(inputCols=['km', 'org_dummy', 'dow_dummy'], outputCol='features')

# Consolidate predictor columns
flites = assembler.transform(flites)

# Check the resulting column
#flites.distinct().show(8, truncate=False)

# Split the data into training and testing sets
flights_train, flights_test = flites.randomSplit([0.8, 0.2], seed=23)
#print(flights_train.toPandas().shape, flights_test.toPandas().shape)

# Create a regression object and train on training data
regression = LinearRegression(labelCol="duration", featuresCol = 'features').fit(flights_train)

# Create predictions for the testing data and take a look at the predictions
predictions = regression.transform(flights_test)
#predictions.select('duration', 'prediction').show(truncate=False)
print(predictions.toPandas().sample(12))

# Calculate the RMSE
print("\nRMSE", RegressionEvaluator(labelCol="duration").evaluate(predictions))

# Print the coefficients and intercept for linear regression
print("\nCoefficients: %s" % str(regression.coefficients))
print("Intercept: %s" % str(regression.intercept))

# Summarize the model over the training set and print out some metrics
trainingSummary = regression.summary
print("numIterations: %d" % trainingSummary.totalIterations)
print("objectiveHistory: %s\n" % str(trainingSummary.objectiveHistory))
#trainingSummary.residuals.show(8)
print("\nRMSE: %f" % trainingSummary.rootMeanSquaredError)
print("r2: %f" % trainingSummary.r2)

# Average speed in km per hour
avg_speed = regression.intercept / regression.coefficients[0]
print("\nAverage speed in km/h", avg_speed)

spark.stop()
'''
RMSE 16.750777486382123

Coefficients: [0.07568797190166142]
Intercept: 44.114933758884426
numIterations: 1
objectiveHistory: [0.0]


RMSE: 17.021932
r2: 0.962509

Average speed in km/h 582.8526336549396

V2
RMSE 10.829636099618703

Coefficients: [0.07427522396601931,29.034645216612304,20.947533062153802,52.843260459398145,47.07820683731689,16.07354745008443,18.61418840829654,18.11170709292439,0.43903177502097956,0.0978389381566636,-0.29580709670633754,0.0814943130635107,0.22633702954660978,-0.07813086413746358]
Intercept: 15.361566402133885
numIterations: 1
objectiveHistory: [0.0]


RMSE: 11.085525
r2: 0.984047

Average speed in km/h 206.81952314491514
'''
