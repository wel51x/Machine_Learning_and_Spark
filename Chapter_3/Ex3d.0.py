from pyspark.sql import SparkSession
from pyspark.sql.functions import round
from pyspark.ml.feature import StringIndexer, OneHotEncoderEstimator, VectorAssembler
from pyspark.mllib.linalg import DenseVector, SparseVector
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

print("Cars with null cyl", cars.filter('cyl IS NULL').count(), '\n')

indexer = StringIndexer(inputCol='type',
                        outputCol='type_idx')

# Assign index values to strings
indexer = indexer.fit(cars)
# Create column with index values
cars = indexer.transform(cars)

cars = cars.withColumn('density', round(cars.weight_kg / cars.length_meters, 2))
cars = cars.withColumn('density_area', round(cars.weight_kg / cars.length_meters**2, 2))
cars = cars.withColumn('density_volume', round(cars.weight_kg / cars.length_meters**3, 2))

onehot = OneHotEncoderEstimator(inputCols=['type_idx'], outputCols=['type_dummy'])
onehot = onehot.fit(cars)
cars = onehot.transform(cars)

pd.set_option('display.max_columns', None) # all cols
pd.set_option('display.width', 161)
pd.set_option('display.max_colwidth', 199)
#print(cars.toPandas().sample(12))

# Check column data types
print('\n', cars.dtypes, '\n')

assembler = VectorAssembler(inputCols=['weight_kg', 'cyl', 'type_dummy', 'density', 'density_area', 'density_volume'],
                            outputCol='features')
cars = assembler.transform(cars)

kars = cars.select('consumption', 'features')

print(kars.toPandas().sample(12))

# Split the data into training and testing sets
kars_train, kars_test = kars.randomSplit([0.8, 0.2], seed=23)

regression = LinearRegression(labelCol='consumption').fit(kars_train)

# Create predictions for the testing data and take a look at the predictions
predictions = regression.transform(kars_test)
print("\nStandard Linear Regression")
#print("\nStandard Linear Regression\nSample")
#print(predictions.toPandas().sample(12))

# Print the coefficients and RMSE for linear regression
trainingSummary = regression.summary
print("Coefficients: %s" % str(regression.coefficients))
print("RMSE: %f" % trainingSummary.rootMeanSquaredError)

# Ridge regression
ridge = LinearRegression(labelCol='consumption', elasticNetParam=0, regParam=0.1).fit(kars_train)
# Create predictions for the testing data and take a look at the predictions
predictions = ridge.transform(kars_test)
print("\nRidge Regression")
#print("\nRidge Regression\nSample")
#print(predictions.toPandas().sample(12))

# Print the coefficients and RMSE for Ridge regression
trainingSummary = ridge.summary
print("Coefficients: %s" % str(ridge.coefficients))
print("RMSE: %f" % trainingSummary.rootMeanSquaredError)

# Lasso regression
# ? = 0.1 | ? = 1 -> Lasso
lasso = LinearRegression(labelCol='consumption', elasticNetParam=1, regParam=0.75).fit(kars_train)
# Create predictions for the testing data and take a look at the predictions
predictions = lasso.transform(kars_test)
print("\nLasso Regression")
#print("\nLasso Regression\nSample")
#print(predictions.toPandas().sample(12))

# Print the coefficients and RMSE for Lasso regression
trainingSummary = lasso.summary
print("Coefficients: %s" % str(lasso.coefficients))
print("RMSE: %f" % trainingSummary.rootMeanSquaredError)

spark.stop()
