import pyspark
from pyspark.sql.functions import round
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

# Creating a SparkSession
spark = SparkSession.builder \
                    .master('local[*]') \
                    .appName('first_spark_application') \
                    .getOrCreate()

#flights = spark.read.csv('/Users/wel51x/Box Sync/MyBox/Code/DataCamp/data/flights.csv',
flights = spark.read.csv('/Users/wel51x/Box Sync/MyBox/Code/DataCamp/data/flights_truncated.csv',
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

# Create an indexer for org categorical feature
flights_indexed = StringIndexer(inputCol="org", outputCol='org_idx').fit(flights).transform(flights)
# Check first five records
#flights_indexed.show(5)

flites = flights_indexed.select('km', 'org_idx', 'duration')

# Create 'features' vector: 'weight_kg', 'cyl', 'type_dummy'
assembler = VectorAssembler(inputCols=['km', 'org_idx'], outputCol='features')

# Consolidate predictor columns
flights_assembled = assembler.transform(flites)

# Check the resulting column
flites = flights_assembled.select('duration', 'features')
#flites.distinct().show(8, truncate=False)

print("Sample model input")
print(flites.toPandas().sample(12))

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
===>>> Only works if one-hot encode org, which I'm too unmotivated to do
# Average speed in km per hour
avg_speed_hour = 60 / regression.coefficients[0]
print(avg_speed_hour)

# Average minutes on ground at OGG
inter = regression.intercept
print(inter)

# Average minutes on ground at JFK
avg_ground_jfk = inter + regression.coefficients[3]
print(avg_ground_jfk)

# Average minutes on ground at LGA
avg_ground_lga = inter + regression.coefficients[4]
print(avg_ground_lga)
'''
