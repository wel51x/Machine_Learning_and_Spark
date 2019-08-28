from pyspark.sql import SparkSession
from pyspark.sql.functions import round
from pyspark.ml.feature import StringIndexer, OneHotEncoderEstimator, VectorAssembler
from pyspark.ml.regression import LinearRegression
import pandas as pd

# Creating a SparkSession
spark = SparkSession.builder \
                    .master('local[*]') \
                    .appName('first_spark_application') \
                    .getOrCreate()

cars = spark.read.csv('/Users/wel51x/Box Sync/MyBox/Code/DataCamp/data/cars.csv',
                         sep=',',
                         header=True,
                         inferSchema=True,
                         nullValue='NA')

cars = cars.dropna()

# Get number of records
print("The data contains %d records." % cars.count(), '\n')

cars = cars.withColumnRenamed("ncyl", "cyl")
cars = cars.withColumn('length_meters', round(cars.length * 0.0254, 3))

cars = cars.withColumn('weight_kg', round(cars.weight / 2.205, 0))

cars = cars.withColumn('avg_mpg', round((cars.city_mpg + cars.hwy_mpg) / 2, 1)) \
            .drop("city_mpg", "hwy_mpg")

cars = cars.withColumn('consumption', round((100 * 3.785411784) / (cars.avg_mpg * 1.609344), 2))

pd.set_option('display.max_columns', None) # all cols
pd.set_option('display.width', 161)
#print(cars.toPandas().sample(8), '\n')

indexer = StringIndexer(inputCol='type',
                        outputCol='type_idx')

# Assign index values to strings
indexer = indexer.fit(cars)
# Create column with index values
cars = indexer.transform(cars)

# Check column data types
print('\n', cars.dtypes, '\n')

kars = cars.select('name', 'weight_kg', 'cyl', 'consumption', 'type', 'type_idx')

#print(kars.toPandas().sample(12))

onehot = OneHotEncoderEstimator(inputCols=['type_idx'], outputCols=['type_dummy'])
onehot = onehot.fit(kars)
kars = onehot.transform(kars)
kars.distinct().show(8, truncate=False)

# Create 'features' vector: 'weight_kg', 'cyl', 'type_dummy'
assembler = VectorAssembler(inputCols=['weight_kg', 'cyl', 'type_dummy'], outputCol='features')

# Consolidate predictor columns
kars_assembled = assembler.transform(kars)

# Check the resulting column
kars = kars_assembled.select('features', 'consumption')
kars.distinct().show(8, truncate=False)

# Split the data into training and testing sets
kars_train, kars_test = kars.randomSplit([0.8, 0.2], seed=23)

# Fit a Logistic Regression model to the training data
regression = LinearRegression(labelCol='consumption')
regression = regression.fit(kars_train)

# Make predictions on the testing data
prediction = regression.transform(kars_test)

# Create a confusion matrix, comparing predictions to known labels
prediction.groupBy("consumption", 'prediction').count().show(8)

# Print the coefficients and intercept for linear regression
print("Coefficients: %s" % str(regression.coefficients))
print("Intercept: %s" % str(regression.intercept))

# Summarize the model over the training set and print out some metrics
trainingSummary = regression.summary
print("numIterations: %d" % trainingSummary.totalIterations)
print("objectiveHistory: %s" % str(trainingSummary.objectiveHistory))
trainingSummary.residuals.show(8)
print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
print("r2: %f" % trainingSummary.r2)

# Find RMSE (Root Mean Squared Error)
#print("RMSE:", RegressionEvaluator(labelCol='consumption').evaluate(prediction))

spark.stop()
