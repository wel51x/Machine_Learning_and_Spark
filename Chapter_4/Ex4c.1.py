# derived from EX4a.1
import pyspark
from pyspark.sql.functions import round
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, OneHotEncoderEstimator, VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
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

# Create parameter grid
params = ParamGridBuilder()
# Add grids for two parameters
params = params.addGrid(regression.regParam, [0.01, 0.1, 1.0, 10.0]) \
               .addGrid(regression.elasticNetParam, [0, 0.5, 1.0])

# Build the parameter grid
params = params.build()

print('Number of models to be tested: ', len(params))

# object to evaluate performance
evaluator = RegressionEvaluator(labelCol='duration')

# create cross-validation object
cv = CrossValidator(estimator=pipeline,
                    estimatorParamMaps=params,
                    evaluator=evaluator,
                    numFolds=5,
                    seed=13)

# run fit on training data
#pipeline = pipeline.fit(flights_train)
cv = cv.fit(flights_train)

# Get the best model from cross validation
best_model = cv.bestModel

# Look at the stages in the best model
print("Best model stages:", best_model.stages)

# Get the parameters for the LinearRegression object in the best model
print("Best model LinReg ParamMap:", best_model.stages[3].extractParamMap())

# Print the RMSE for folds and evaluator
print("Average RMSE across all folds: %f" % cv.avgMetrics[0])

# Make predictions on the testing data
print("Best Model RMSE: %f" % evaluator.evaluate(best_model.transform(flights_test)))

spark.stop()
'''
Pipeline
RMSE 10.829636099618703

Coefficients: [0.07427522396601931,29.03464521661228,20.94753306215377,52.843260459398124,47.07820683731686,18.61418840829651,16.07354745008441,18.111707092924366,0.4390317750209789,0.09783893815666257,-0.2958070967063388,0.0814943130635097,0.226337029546609,-0.0781308641374644]
Intercept: 15.361566402133912
numIterations: 1
objectiveHistory: [0.0]


RMSE: 11.085525
r2: 0.984047

Average speed in km/h 206.81952314491548

V2
Average RMSE across all folds: 11.091014
evaluator RMSE: 10.829636
'''
