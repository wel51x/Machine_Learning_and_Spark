from pyspark.sql import SparkSession
from pyspark.sql.functions import round
from pyspark.ml.feature import StringIndexer, OneHotEncoderEstimator, VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
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

kars = cars.select('weight_kg', 'cyl', 'consumption', 'type')

# OneHot encode type
onehot = OneHotEncoderEstimator(inputCols=['type_idx'], outputCols=['type_dummy'])

# Create 'features' vector: 'weight_kg', 'cyl', 'type_dummy'
assembler = VectorAssembler(inputCols=['weight_kg', 'cyl', 'type_dummy'], outputCol='features')

# Split the data into training and testing sets
kars_train, kars_test = kars.randomSplit([0.8, 0.2], seed=23)

# Fit a Logistic Regression model to the training data
regression = LinearRegression(labelCol='consumption')

# Combine steps into a pipeline
pipeline = Pipeline(stages=[indexer, onehot, assembler, regression])

# object to evaluate performance
evaluator = RegressionEvaluator(labelCol='consumption')

# build grid of parameter values (now empty)
params = ParamGridBuilder().build()

# create cross-validation object
cv = CrossValidator(estimator=pipeline,
                    estimatorParamMaps=params,
                    evaluator=evaluator,
                    numFolds=10,
                    seed=13)

# run fit on training data
#pipeline = pipeline.fit(kars_train)
cv = cv.fit(kars_train)
print("Average RMSE across all folds: %f" % cv.avgMetrics[0])

# Make predictions on the testing data
print("evaluator RMSE: %f" % evaluator.evaluate(cv.transform(kars_test)))

spark.stop()
