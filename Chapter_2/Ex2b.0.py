import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import round
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
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
                        outputCol='origin_idx')

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
kars = cars_assembled.select('features', 'origin_idx')
#kars.show(9)

# Split data into training and testing sets
kars_train, kars_test = kars.randomSplit([0.8, 0.2], seed=23)

print(kars_train.count(), kars_test.count())

# Create a Decision Tree classifier
tree = DecisionTreeClassifier(labelCol="origin_idx")

# Learn from training data
tree = tree.fit(kars_train)

# Make predictions on testing data
prediction = tree.transform(kars_test)

prediction.show(9)

# Confusion matrix
confusion_matrix = prediction.groupBy("origin_idx", "prediction").count()
confusion_matrix.show()

# Accuracy
evaluator = MulticlassClassificationEvaluator(labelCol="origin_idx", metricName="accuracy")
accuracy = evaluator.evaluate(prediction)
print("Test set accuracy = " + str(accuracy))

# Find weighted precision
multi_evaluator = MulticlassClassificationEvaluator(labelCol="origin_idx")
accuracy = multi_evaluator.evaluate(prediction, {multi_evaluator.metricName: "accuracy"})
weighted_precision = multi_evaluator.evaluate(prediction, {multi_evaluator.metricName: "weightedPrecision"})
weighted_recall = multi_evaluator.evaluate(prediction, {multi_evaluator.metricName: "weightedRecall"})

# Find AUC
binary_evaluator = BinaryClassificationEvaluator(labelCol="origin_idx")
auc = binary_evaluator.evaluate(prediction, {binary_evaluator.metricName: "areaUnderROC"})
print("Summary Stats")
print('accuracy = {:.3f}\nweighted precision = {:.3f}'.format(accuracy, weighted_precision))
print('weighted recall = {:.3f}\nareaUnderROC = {:.3f}'.format(weighted_recall, auc))

spark.stop()

'''
+----------+----------+-----+
|origin_idx|prediction|count|
+----------+----------+-----+
|       1.0|       1.0|   21|
|       0.0|       1.0|    9|
|       1.0|       0.0|    7|
|       0.0|       0.0|   49|
+----------+----------+-----+

Test set accuracy = 0.813953488372093
'''
