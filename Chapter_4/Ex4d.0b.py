# Gradient-Boosted Tree Classifier - derived from Ex2b.0
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import round
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import GBTClassifier
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

cars = cars.withColumn('consumption', round((100 * 3.785411784) / (cars.avg_mpg * 1.609344), 2))

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
assembler = VectorAssembler(inputCols=['eng_size', 'cyl', 'consumption', 'length_meters', 'weight_kg'],
                            outputCol='features')

# Consolidate predictor columns
cars_assembled = assembler.transform(cars)

# Check the resulting column
kars = cars_assembled.select('features', 'origin_idx')
#kars.show(9)

# Split data into training and testing sets
kars_train, kars_test = kars.randomSplit([0.8, 0.2], seed=23)

print(kars_train.count(), kars_test.count())

# Create a Gradient-Boosted Tree classifier
#tree = DecisionTreeClassifier(labelCol="origin_idx")
#forest = RandomForestClassifier(labelCol="origin_idx",
#                                numTrees=5)
gbt = GBTClassifier(labelCol="origin_idx",
                    maxIter=10)

# Learn from training data
#tree = tree.fit(kars_train)
#tree = forest.fit(kars_train)
gbt = gbt.fit(kars_train)

# Make predictions on testing data
#prediction = tree.transform(kars_test)
#prediction = forest.transform(kars_test)
prediction = gbt.transform(kars_test)

prediction.show(9, False)

print("\nforest.featureImportances:", gbt.featureImportances,'\n')
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
==>>Decision Tree Classifier
+----------+----------+-----+
|origin_idx|prediction|count|
+----------+----------+-----+
|       1.0|       1.0|   21|
|       0.0|       1.0|    9|
|       1.0|       0.0|    7|
|       0.0|       0.0|   49|
+----------+----------+-----+

Test set accuracy = 0.813953488372093
areaUnderROC = 0.468

==>>Random Forest Classifier:
forest.trees:
  DecisionTreeClassificationModel (uid=dtc_e61fa9bf769d) of depth 5 with 33 nodes
  DecisionTreeClassificationModel (uid=dtc_be745c317ca6) of depth 5 with 41 nodes
  DecisionTreeClassificationModel (uid=dtc_79d6bac1bbb0) of depth 5 with 37 nodes
  DecisionTreeClassificationModel (uid=dtc_1a2d1d7b2d39) of depth 5 with 35 nodes
  DecisionTreeClassificationModel (uid=dtc_fb06e18a2785) of depth 5 with 31 nodes

                                            'eng_size', 'cyl', 'consumption', 'length_meters', 'weight_kg'
forest.featureImportances: (5,[0,1,2,3,4],[0.23237547710500045,0.0238418182566662,0.14416803677293258,0.48937637053472194,0.11023829733067884]) 

+----------+----------+-----+
|origin_idx|prediction|count|
+----------+----------+-----+
|       1.0|       1.0|   17|
|       0.0|       1.0|    6|
|       1.0|       0.0|    9|
|       0.0|       0.0|   54|
+----------+----------+-----+

Test set accuracy = 0.8255813953488372

==>>Random Forest Classifier (second run):
forest.featureImportances: (5,[0,1,2,3,4],[0.3496815892465469,0.04635808587962548,0.10974057779030717,0.3734804397303805,0.12073930735313998]) 

+----------+----------+-----+
|origin_idx|prediction|count|
+----------+----------+-----+
|       1.0|       1.0|   19|
|       0.0|       1.0|    9|
|       1.0|       0.0|    7|
|       0.0|       0.0|   51|
+----------+----------+-----+

Test set accuracy = 0.813953488372093

==>>Random Forest Classifier (third run):

forest.featureImportances: (5,[0,1,2,3,4],[0.19515730727424616,0.014998024420923933,0.14716092635277428,0.416869031026445,0.22581471092561056]) 

+----------+----------+-----+
|origin_idx|prediction|count|
+----------+----------+-----+
|       1.0|       1.0|   18|
|       0.0|       1.0|    5|
|       1.0|       0.0|    8|
|       0.0|       0.0|   55|
+----------+----------+-----+

Test set accuracy = 0.8488372093023255
areaUnderROC = 0.879

==>>Gradient-Boosted Tree Classifier

                                            'eng_size', 'cyl', 'consumption', 'length_meters', 'weight_kg'
forest.featureImportances: (5,[0,1,2,3,4],[0.2984863323323187,0.04268468662676176,0.2204502496975831,0.29914260582153573,0.13923612552180065]) 

+----------+----------+-----+
|origin_idx|prediction|count|
+----------+----------+-----+
|       1.0|       1.0|   23|
|       0.0|       1.0|    5|
|       1.0|       0.0|    3|
|       0.0|       0.0|   55|
+----------+----------+-----+

Test set accuracy = 0.9069767441860465
areaUnderROC = 0.943
'''
