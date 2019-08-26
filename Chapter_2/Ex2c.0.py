import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import round
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
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

print("Cars with null cyl", cars.filter('cyl IS NULL').count(), '\n')

indexer = StringIndexer(inputCol='origin',
                        outputCol='label')

# Assign index values to strings
indexer = indexer.fit(cars)
# Create column with index values
cars = indexer.transform(cars)

pd.set_option('display.max_columns', None) # all cols
pd.set_option('display.width', 161)
print(cars.toPandas().sample(12))

print(indexer)
# View the first five records
cars.sample(False, .25).show()

# Check column data types
print('\n', cars.dtypes, '\n')

#cars.printSchema()

#print("\nNull data exists:", pd.isnull(cars), '\n')

# Create 'features' vector: 'eng_size', 'cyl', 'avg_mpg', 'wheel_base', 'weight'
assembler = VectorAssembler(inputCols=['eng_size', 'cyl', 'avg_mpg', 'wheel_base', 'weight'], outputCol='features')

# Consolidate predictor columns
cars_assembled = assembler.transform(cars)

# Check the resulting column
kars = cars_assembled.select('features', 'label')
#kars.show(9)

# Split data into training and testing sets
kars_train, kars_test = kars.randomSplit([0.8, 0.2], seed=23)

print(kars_train.count(), kars_test.count())

# Create a Decision Tree classifier
model = LogisticRegression()

# Learn from training data
model_fit = model.fit(kars_train)

# Make predictions on testing data
prediction = model_fit.transform(kars_test)

prediction.show(9)

# Confusion matrix
confusion_matrix = prediction.groupBy("label", "prediction").count()
confusion_matrix.show()

# Calculate the elements of the confusion matrix
TrueNeg = prediction.filter('prediction = 0 AND label = prediction').count()
TruePos = prediction.filter('prediction = 1 AND label = prediction').count()
FalseNeg = prediction.filter('prediction = 0 AND label = 1').count()
FalsePos = prediction.filter('prediction = 1 AND label = 0').count()

# Accuracy measures the proportion of correct predictions
print("Summary Stats")
accuracy = (TrueNeg + TruePos) / (TrueNeg + TruePos + FalseNeg + FalsePos)
print("Accuracy:", accuracy)
precision = TruePos / (TruePos + FalsePos)
recall = TruePos / (TruePos + FalseNeg)
print("Precision:", precision)
print("Recall:", recall)

evaluator = MulticlassClassificationEvaluator()
accuracy = evaluator.evaluate(prediction, {evaluator.metricName: 'accuracy'})
precision = evaluator.evaluate(prediction, {evaluator.metricName: 'weightedPrecision'})
recall = evaluator.evaluate(prediction, {evaluator.metricName: 'weightedRecall'})
print("MulticlassClassificationEvaluator Stats")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)

spark.stop()

'''
+----------+----------+-----+
|origin_idx|prediction|count|
+----------+----------+-----+
|       1.0|       1.0|   12|
|       0.0|       1.0|    3|
|       1.0|       0.0|   16|
|       0.0|       0.0|   55|
+----------+----------+-----+

Test set accuracy = 0.7790697674418605
Test set accuracy = 0.813953488372093 (Decision Tree)
'''
