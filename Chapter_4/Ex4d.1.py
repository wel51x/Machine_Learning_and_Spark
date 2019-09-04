# derived from EX4a.1
import pyspark
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier, GBTClassifier
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

# Create 'features' vector: 'km', 'org_dummy', 'dow_dummy'
assembler = VectorAssembler(inputCols=['mon', 'depart', 'duration'], outputCol='features')

# Consolidate predictor columns
flites = assembler.transform(flites)

pd.set_option('display.max_columns', None) # all cols
pd.set_option('display.width', 161)
pd.set_option('display.max_colwidth', 199)

print("Sample model input")
print(flites.toPandas().sample(12))

# Split the data into training and testing sets
flights_train, flights_test = flites.randomSplit([0.8, 0.2], seed=23)

# Create model objects and train on training data
tree = DecisionTreeClassifier().fit(flights_train)
gbt = GBTClassifier().fit(flights_train)

# Compare AUC on testing data
evaluator = BinaryClassificationEvaluator()
evaluator.evaluate(tree.transform(flights_test))
evaluator.evaluate(gbt.transform(flights_test))

# Find the number of trees and the relative importance of features
print(tree._call_java('toDebugString'))
'''
print("\nDecisionTreeClassifier trees:")
for i in tree.trees:
    print(" ", i)
print(tree.featureImportances)
print()
'''
#print(gbt.trees)
print("\nGBTClassifier trees:")
for i in gbt.trees:
    print(" ", i)
print()
print(gbt.featureImportances)

spark.stop()
'''
DecisionTreeClassificationModel (uid=DecisionTreeClassifier_a6d8645d598e) of depth 5 with 29 nodes
  If (feature 1 <= 9.524999999999999)
   If (feature 2 <= 119.5)
    Predict: 0.0
   Else (feature 2 > 119.5)
    If (feature 0 <= 6.5)
     If (feature 1 <= 7.74)
      Predict: 0.0
     Else (feature 1 > 7.74)
      If (feature 0 <= 4.5)
       Predict: 0.0
      Else (feature 0 > 4.5)
       Predict: 1.0
    Else (feature 0 > 6.5)
     If (feature 0 <= 10.5)
      Predict: 0.0
     Else (feature 0 > 10.5)
      If (feature 1 <= 8.59)
       Predict: 0.0
      Else (feature 1 > 8.59)
       Predict: 1.0
  Else (feature 1 > 9.524999999999999)
   If (feature 0 <= 6.5)
    If (feature 2 <= 125.5)
     If (feature 2 <= 50.5)
      Predict: 0.0
     Else (feature 2 > 50.5)
      If (feature 1 <= 13.425)
       Predict: 0.0
      Else (feature 1 > 13.425)
       Predict: 1.0
    Else (feature 2 > 125.5)
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
  DecisionTreeRegressionModel (uid=dtr_9719243683f8) of depth 5 with 63 nodes
  DecisionTreeRegressionModel (uid=dtr_dc01c669ecef) of depth 5 with 63 nodes
  DecisionTreeRegressionModel (uid=dtr_44ada9034bb7) of depth 5 with 63 nodes
  DecisionTreeRegressionModel (uid=dtr_993ccc594474) of depth 5 with 63 nodes
  DecisionTreeRegressionModel (uid=dtr_67380a052abe) of depth 5 with 63 nodes
  DecisionTreeRegressionModel (uid=dtr_cffc5fcf5cb9) of depth 5 with 63 nodes
  DecisionTreeRegressionModel (uid=dtr_d75aa1c8742d) of depth 5 with 63 nodes
  DecisionTreeRegressionModel (uid=dtr_4e74be8bcb14) of depth 5 with 63 nodes
  DecisionTreeRegressionModel (uid=dtr_99831d64c136) of depth 5 with 63 nodes
  DecisionTreeRegressionModel (uid=dtr_f16ca0c65b1e) of depth 5 with 63 nodes
  DecisionTreeRegressionModel (uid=dtr_eb8368dd3eb6) of depth 5 with 61 nodes
  DecisionTreeRegressionModel (uid=dtr_5575d25f59c3) of depth 5 with 63 nodes
  DecisionTreeRegressionModel (uid=dtr_bc9af10f2078) of depth 5 with 63 nodes
  DecisionTreeRegressionModel (uid=dtr_83dbb602ee08) of depth 5 with 61 nodes
  DecisionTreeRegressionModel (uid=dtr_96c0923c2784) of depth 5 with 63 nodes
  DecisionTreeRegressionModel (uid=dtr_d052e42b6847) of depth 5 with 63 nodes
  DecisionTreeRegressionModel (uid=dtr_278debbd0005) of depth 5 with 63 nodes
  DecisionTreeRegressionModel (uid=dtr_9b3bc7c606f8) of depth 5 with 63 nodes
  DecisionTreeRegressionModel (uid=dtr_a262423f1e37) of depth 5 with 63 nodes
  DecisionTreeRegressionModel (uid=dtr_f2433aa1ef0c) of depth 5 with 63 nodes

(3,[0,1,2],[0.28006563889404096,0.3193787093092136,0.40055565179674546])
'''
