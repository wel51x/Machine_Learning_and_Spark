from pyspark.sql import SparkSession
from pyspark.sql.functions import round
from pyspark.ml.feature import StringIndexer, OneHotEncoderEstimator, VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline
import pandas as pd

REGRESSION_STAGE = 3

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
print("The data contains %d records." % cars.count())

cars = cars.withColumnRenamed("ncyl", "cyl")
cars = cars.withColumn('length_meters', round(cars.length * 0.0254, 3))

cars = cars.withColumn('weight_kg', round(cars.weight / 2.205, 0))

cars = cars.withColumn('avg_mpg', round((cars.city_mpg + cars.hwy_mpg) / 2, 1)) \
            .drop("city_mpg", "hwy_mpg")

cars = cars.withColumn('consumption', round((100 * 3.785411784) / (cars.avg_mpg * 1.609344), 2))

pd.set_option('display.max_columns', None) # all cols
pd.set_option('display.width', 161)
pd.set_option('display.max_colwidth', 199)

# Create indexer
indexer = StringIndexer(inputCol='type',
                        outputCol='type_idx')

# Check column data types
print('\n', cars.dtypes, '\n')

kars = cars.select('name', 'weight_kg', 'cyl', 'consumption', 'type')

onehot = OneHotEncoderEstimator(inputCols=['type_idx'], outputCols=['type_dummy'])

# Create 'features' vector: 'weight_kg', 'cyl', 'type_dummy'
assembler = VectorAssembler(inputCols=['weight_kg', 'cyl', 'type_dummy'], outputCol='features')

# Split the data into training and testing sets
kars_train, kars_test = kars.randomSplit([0.8, 0.2], seed=23)

# Fit a Logistic Regression model to the training data
regression = LinearRegression(labelCol='consumption')

# Combine steps into a pipeline
pipeline = Pipeline(stages=[indexer, onehot, assembler, regression])

# run fit on training data
pipeline = pipeline.fit(kars_train)

# Make predictions on the testing data
prediction = pipeline.transform(kars_test)

# Create a confusion matrix, comparing predictions to known labels
prediction.groupBy("consumption", 'prediction').count().show(8)

# Print the coefficients and intercept for linear regression
print("Coefficients: %s" % str(pipeline.stages[REGRESSION_STAGE].coefficients))
print("Intercept: %s" % str(pipeline.stages[REGRESSION_STAGE].intercept))

# Summarize the model over the training set and print out some metrics
trainingSummary = pipeline.stages[REGRESSION_STAGE].summary
print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
print("r2: %f" % trainingSummary.r2)

spark.stop()
'''
+-----------+------------------+-----+
|consumption|        prediction|count|
+-----------+------------------+-----+
|      10.94|11.321502227623656|    1|
|      10.45| 11.50917390307143|    1|
|       9.41| 9.637574456512294|    1|
|       7.47| 7.693532181917655|    1|
|       11.2|10.903093161784035|    1|
|       6.53| 7.491412346760725|    1|
|      12.06|12.144401554398481|    1|
|      13.07|11.614263592344276|    1|
+-----------+------------------+-----+

Coefficients: [0.004370804199346645,0.22951584440188164,-0.057988608954411255,0.055685278360261874,0.9700456271308224,1.0982599913336637,0.5461707604103592,-0.08315059216350616]
Intercept: 1.670474811849168

RMSE: 0.877274
r2: 0.826280
'''
