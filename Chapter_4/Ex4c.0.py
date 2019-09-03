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

# Check column data types
print('\n', cars.dtypes, '\n')

kars = cars.select('weight_kg', 'cyl', 'consumption', 'type')

# Create indexer
indexer = StringIndexer(inputCol='type', outputCol='type_idx')

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

# build grid of parameter values
params = ParamGridBuilder()
params = params.addGrid(regression.fitIntercept, [True, False])\
               .addGrid(regression.regParam, [0.001, 0.01, 0.1, 1, 10]) \
               .addGrid(regression.elasticNetParam, [0, 0.25, 0.5, 0.75, 1])

params = params.build()

print ('\nNumber of models to be tested: ', len(params))

# create cross-validation object
cv = CrossValidator(estimator=pipeline,
                    estimatorParamMaps=params,
                    evaluator=evaluator)
cv = cv.setNumFolds(10).setSeed(13)

# run fit on training data
#pipeline = pipeline.fit(kars_train)
cv = cv.fit(kars_train)
print("Average RMSE across all folds:", cv.avgMetrics)

# Best Model
best_mod = cv.bestModel
print("Best Model:", best_mod)

# Make predictions on the testing data
print("evaluator RMSE: %f" % evaluator.evaluate(best_mod.transform(kars_test)))

param_dict = best_mod.stages[-1].extractParamMap()

sane_dict = {}
for k, v in param_dict.items():
    print(k.name, "=", v)
    sane_dict[k.name] = v

best_reg = sane_dict["regParam"]
best_elastic_net = sane_dict["elasticNetParam"]
best_max_iter = sane_dict["maxIter"]

print('Best Param (regParam): ', best_reg)
print('Best Param (MaxIter): ', best_max_iter)
print('Best Param (elasticNetParam): ', best_elastic_net)

spark.stop()
'''
Average RMSE across all folds: [0.8884161396108119, 0.8885570162342967, 0.8887003373615797, 0.8888460854480509,
 0.8889469112883204, 0.888638790371932, 0.8896265895818797, 0.8907399503625433, 0.8919003208219313,
 0.8932097339974567, 0.894205188615833, 0.907306467005484, 0.9165213571326879, 0.9225497645230731,
 0.9290233961412238, 0.9958342845748357, 1.0831470869945592, 1.1994950096532535, 1.3145922656780331,
 1.4190054287185188, 1.6126334128475643, 2.1708216814479897, 2.1708216814479897, 2.1708216814479897,
 2.1708216814479897, 0.9590668092869994, 0.9598271610930321, 0.9609384803051157, 0.9621303104254229,
 0.9616762164509155, 0.9590497701895473, 0.9586935312415451, 0.9584540186309892, 0.9575686306409834,
 0.9571689032652976, 0.9604868497536181, 0.9575168821871982, 0.9600267570973535, 0.9627073856626984,
 0.967787821542842, 0.995212985975225, 1.0142016366824742, 1.0149616270114779, 1.0147099412354486,
 1.0554103745652683, 1.481761936331171, 1.6900082994947436, 1.898409984381161, 2.1412922880403, 2.2646836098809184]
Best Model: PipelineModel_467a621a6732
evaluator RMSE: 0.809642
aggregationDepth = 2
elasticNetParam = 0.0
epsilon = 1.35
featuresCol = features
fitIntercept = True
labelCol = consumption
loss = squaredError
maxIter = 100
predictionCol = prediction
regParam = 0.001
solver = auto
standardization = True
tol = 1e-06
Best Param (regParam):  0.001
Best Param (MaxIter):  100
Best Param (elasticNetParam):  0.0
'''
