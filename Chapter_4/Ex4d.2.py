# derived from EX4a.1
import pyspark
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator
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

# Remove records with missing values in any column and get the number of remaining rows
flights = flights.dropna()
print("The data contains %d records after dropping records with na values." % flights.count())

# Create 'label' column indicating whether flight delayed (1) or not (0)
flights = flights.withColumn('label', (flights.delay >= 15).cast('integer'))

flites = flights.select('mon', 'depart', 'duration', 'label')

pd.set_option('display.max_columns', None) # all cols
pd.set_option('display.width', 161)
pd.set_option('display.max_colwidth', 199)

# Create 'features' vector: 'km', 'org_dummy', 'dow_dummy'
assembler = VectorAssembler(inputCols=['mon', 'depart', 'duration'], outputCol='features')

# Consolidate predictor columns
flites = assembler.transform(flites)

print("Sample model input")
print(flites.toPandas().sample(12))

# Split the data into training and testing sets
flights_train, flights_test = flites.randomSplit([0.8, 0.2], seed=23)

# Create model objects and train on training data
#tree = DecisionTreeClassifier().fit(flights_train)
#gbt = GBTClassifier().fit(flights_train)

forest = RandomForestClassifier()

# Create parameter grid
params = ParamGridBuilder()
# Add grids for two parameters
params = params.addGrid(forest.featureSubsetStrategy, ['all', 'onethird', 'sqrt', 'log2']) \
               .addGrid(forest.maxDepth, [2, 5, 10])

# Build the parameter grid
params = params.build()

# Compare AUC on testing data
evaluator = BinaryClassificationEvaluator()

# create cross-validation object
cv = CrossValidator(estimator=forest,
                    estimatorParamMaps=params,
                    evaluator=evaluator,
                    numFolds=5,
                    seed=13)

# run fit on training data
cv = cv.fit(flights_train)

# Get the best model from cross validation
best_model = cv.bestModel

# Look at the stages in the best model
print("Best model:", best_model)

# Get the parameters for the LinearRegression object in the best model
print("Best model RandomForestClassifier ParamMap:")
for k, v in best_model.extractParamMap().items():
    print(" ", k.name, "=", v)

# Print the RMSE for folds and evaluator
print("Average RMSE across all folds:", cv.avgMetrics)

# Make predictions on the testing data
print("Best Model RMSE: %f" % evaluator.evaluate(best_model.transform(flights_test)))

# Average AUC for each parameter combination in grid
avg_auc = cv.avgMetrics

# Average AUC for the best model
best_model_auc = max(cv.avgMetrics)

# What's the optimal parameter value?
opt_max_depth = cv.bestModel.explainParam('maxDepth')
opt_feat_substrat = cv.bestModel.explainParam('featureSubsetStrategy')

# AUC for best model on testing data
best_auc = evaluator.evaluate(cv.transform(flights_test))

print("avg_auc:", avg_auc)
print("best_model_auc:", best_model_auc)
print("opt_max_depth:", opt_max_depth)
print("opt_feat_substrat:", opt_feat_substrat)
print("best_auc:", best_auc)

spark.stop()
'''
DecisionTreeClassificationModel (uid=DecisionTreeClassifier_f233574f0321) of depth 5 with 25 nodes
  If (feature 1 <= 9.51)
   If (feature 2 <= 118.5)
    Predict: 0.0
   Else (feature 2 > 118.5)
    If (feature 0 <= 6.5)
     Predict: 0.0
    Else (feature 0 > 6.5)
     If (feature 0 <= 10.5)
      Predict: 0.0
     Else (feature 0 > 10.5)
      If (feature 1 <= 8.575)
       Predict: 0.0
      Else (feature 1 > 8.575)
       Predict: 1.0
  Else (feature 1 > 9.51)
   If (feature 0 <= 6.5)
    If (feature 2 <= 85.5)
     If (feature 2 <= 49.5)
      Predict: 0.0
     Else (feature 2 > 49.5)
      If (feature 0 <= 1.5)
       Predict: 1.0
      Else (feature 0 > 1.5)
       Predict: 0.0
    Else (feature 2 > 85.5)
     Predict: 1.0
   Else (feature 0 > 6.5)
    If (feature 0 <= 10.5)
     If (feature 0 <= 8.5)
      If (feature 2 <= 85.5)
       Predict: 0.0
      Else (feature 2 > 85.5)
       Predict: 1.0
     Else (feature 0 > 8.5)
      Predict: 0.0
    Else (feature 0 > 10.5)
     Predict: 1.0


GBTClassifier trees:
  DecisionTreeRegressionModel (uid=dtr_533c02f10bda) of depth 5 with 63 nodes
  DecisionTreeRegressionModel (uid=dtr_7efa156fe252) of depth 5 with 63 nodes
  DecisionTreeRegressionModel (uid=dtr_7cf20e0c20d9) of depth 5 with 63 nodes
  DecisionTreeRegressionModel (uid=dtr_16ae23b5f16c) of depth 5 with 63 nodes
  DecisionTreeRegressionModel (uid=dtr_49e5065484bd) of depth 5 with 63 nodes
  DecisionTreeRegressionModel (uid=dtr_a64c7391608a) of depth 5 with 63 nodes
  DecisionTreeRegressionModel (uid=dtr_70b8fb8c5fba) of depth 5 with 63 nodes
  DecisionTreeRegressionModel (uid=dtr_fba2bfa015fe) of depth 5 with 63 nodes
  DecisionTreeRegressionModel (uid=dtr_8f059b176a23) of depth 5 with 63 nodes
  DecisionTreeRegressionModel (uid=dtr_4e2bbd10a134) of depth 5 with 63 nodes
  DecisionTreeRegressionModel (uid=dtr_58c117964c8d) of depth 5 with 61 nodes
  DecisionTreeRegressionModel (uid=dtr_13ebf5c1d7eb) of depth 5 with 63 nodes
  DecisionTreeRegressionModel (uid=dtr_09792b3b467f) of depth 5 with 63 nodes
  DecisionTreeRegressionModel (uid=dtr_de4e775cb2be) of depth 5 with 63 nodes
  DecisionTreeRegressionModel (uid=dtr_59caf2357af1) of depth 5 with 61 nodes
  DecisionTreeRegressionModel (uid=dtr_3ec9780243eb) of depth 5 with 63 nodes
  DecisionTreeRegressionModel (uid=dtr_39ca4a1698d9) of depth 5 with 63 nodes
  DecisionTreeRegressionModel (uid=dtr_796f58b035b6) of depth 5 with 61 nodes
  DecisionTreeRegressionModel (uid=dtr_dbb13ed46192) of depth 5 with 63 nodes
  DecisionTreeRegressionModel (uid=dtr_5e1e9e2d3553) of depth 5 with 63 nodes

(3,[0,1,2],[0.332295623201302,0.2837268590362885,0.3839775177624095])
'''
