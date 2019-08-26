import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import round
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
import pandas as pd

# Creating a SparkSession
spark = SparkSession.builder \
                    .master('local[*]') \
                    .appName('first_spark_application') \
                    .getOrCreate()

flights = spark.read.csv('/Users/wel51x/Box Sync/MyBox/Code/DataCamp/data/flights.csv',
                         sep=',',
                         header=True,
                         inferSchema=True,
                         nullValue='NA')

# Get number of records
print("The data contain %d records." % flights.count(), '\n')

# Remove the 'flight' column
flights = flights.drop('flight')

# Number of records with missing 'delay' values
print("Flights with no value in delay field:", flights.filter('delay IS NULL').count())

# Remove records with missing 'delay' values
flights = flights.filter('delay IS NOT NULL')

# Remove records with missing values in any column and get the number of remaining rows
flights = flights.dropna()
print("\nThe data contains %d records after dropping records with na values." % flights.count())

# Convert 'mile' to 'km' and drop 'mile' column
flights = flights.withColumn('km', round(flights.mile * 1.60934, 0)) \
                 .drop('mile')

# Create 'label' column indicating whether flight delayed (1) or not (0)
flights = flights.withColumn('xdelay', (flights.delay >= 15).cast('integer'))

# Create an indexer for carrier categorical feature
indexer = StringIndexer(inputCol="carrier", outputCol='carrier_idx')

# Indexer identifies categories in the data
indexer_model = indexer.fit(flights)

# Indexer creates a new column with numeric index values
flights = indexer_model.transform(flights)

# Repeat the process for the org categorical feature
flights = StringIndexer(inputCol="org", outputCol='org_idx').fit(flights).transform(flights)

# Check first five records
flights.show(5)

# Create an assembler object
assembler = VectorAssembler(inputCols=["mon", "dom", "dow", "carrier_idx", "org_idx", "km", "depart", "duration"],
                            outputCol='features')

# Consolidate predictor columns
flights_assembled = assembler.transform(flights)

# Check the resulting column
flights = flights_assembled.select('features', 'xdelay')

# Split into training and testing sets in a 80:20 ratio
flights_train, flights_test = flights.randomSplit([0.8, 0.2], seed=23)

# Create a classifier object and fit to the training data
tree = LogisticRegression(labelCol="xdelay")
tree_model = tree.fit(flights_train)

# Create predictions for the testing data and take a look at the predictions
prediction = tree_model.transform(flights_test)
predictions = prediction.select('xdelay', 'prediction', 'probability')

print(predictions.toPandas().sample(12))
print()

# Create a confusion matrix
confusion_matrix = prediction.groupBy("xdelay", 'prediction').count()
confusion_matrix.show()

# Calculate the elements of the confusion matrix
TrueNeg = prediction.filter('prediction = 0 AND xdelay = prediction').count()
TruePos = prediction.filter('prediction = 1 AND xdelay = prediction').count()
FalseNeg = prediction.filter('prediction = 0 AND xdelay = 1').count()
FalsePos = prediction.filter('prediction = 1 AND xdelay = 0').count()

# Accuracy measures the proportion of correct predictions
accuracy = (TrueNeg + TruePos) / (TrueNeg + TruePos + FalseNeg + FalsePos)
print("Accuracy", accuracy)
precision = TruePos / (TruePos + FalsePos)
recall = TruePos / (TruePos + FalseNeg)
print('precision = {:.3f}\nrecall    = {:.3f}'.format(precision, recall))

# Find weighted precision
multi_evaluator = MulticlassClassificationEvaluator(labelCol="xdelay")
weighted_precision = multi_evaluator.evaluate(prediction, {multi_evaluator.metricName: "weightedPrecision"})

# Find AUC
binary_evaluator = BinaryClassificationEvaluator(labelCol="xdelay")
auc = binary_evaluator.evaluate(prediction, {binary_evaluator.metricName: "areaUnderROC"})
print('\nweighted prec. = {:.3f}\nareaUnderROC   = {:.3f}'.format(weighted_precision, auc))

spark.stop()
