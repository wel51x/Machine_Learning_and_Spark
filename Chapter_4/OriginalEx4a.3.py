import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import regexp_replace
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
import pandas as pd

# Creating a SparkSession
spark = SparkSession.builder \
                    .master('local[*]') \
                    .appName('first_spark_application') \
                    .getOrCreate()

sms = spark.read.csv('/Users/wel51x/Box Sync/MyBox/Code/DataCamp/data/SMS.csv',
                         sep=';',
                         header=True,
                         inferSchema=True,
                         nullValue='NA')

# Get number of records
print("The data contain %d records." % sms.count(), '\n')

# Check column data types
print('\n', sms.dtypes, '\n')

sms.printSchema()

# Remove punctuation (REGEX provided) and numbers
wrangled = sms.withColumn('text', regexp_replace(sms.text, '[_():;,.!?\\-]', ' '))
wrangled = wrangled.withColumn('text', regexp_replace(wrangled.text, '[0-9]', ' '))

# Merge multiple spaces
wrangled = wrangled.withColumn('text', regexp_replace(wrangled.text, ' +', ' '))

# Text to tokens
tokenizer = Tokenizer(inputCol="text", outputCol="words")
tokenized = tokenizer.transform(wrangled)

# Remove stop words.
remover = StopWordsRemover(inputCol="words", outputCol="terms")
removed = remover.transform(tokenized)

# Apply the hashing trick
hasher = HashingTF(inputCol="terms", outputCol="hash", numFeatures=1024)
hashed = hasher.transform(removed)

# Convert hashed symbols to TF-IDF
idf = IDF(inputCol="hash", outputCol="features")
sms = idf.fit(hashed).transform(hashed)

# View the first four records
sms.show(4, truncate=False)

# Split the data into training and testing sets
sms_train, sms_test = sms.randomSplit([0.8, 0.2], seed=13)

# Fit a Logistic Regression model to the training data
logistic = LogisticRegression(regParam=0.2)
logistic = logistic.fit(sms_train)

# Make predictions on the testing data
prediction = logistic.transform(sms_test)

# Create a confusion matrix, comparing predictions to known labels
prediction.groupBy("label", 'prediction').count().show()

# Find weighted precision
multi_evaluator = MulticlassClassificationEvaluator()
accuracy = multi_evaluator.evaluate(prediction, {multi_evaluator.metricName: "accuracy"})
weighted_precision = multi_evaluator.evaluate(prediction, {multi_evaluator.metricName: "weightedPrecision"})
weighted_recall = multi_evaluator.evaluate(prediction, {multi_evaluator.metricName: "weightedRecall"})

# Find AUC
binary_evaluator = BinaryClassificationEvaluator()
auc = binary_evaluator.evaluate(prediction, {binary_evaluator.metricName: "areaUnderROC"})
print("Summary Stats")
print('accuracy = {:.3f}\nweighted precision = {:.3f}'.format(accuracy, weighted_precision))
print('weighted recall = {:.3f}\nareaUnderROC = {:.3f}'.format(weighted_recall, auc))

spark.stop()
'''
+-----+----------+-----+
|label|prediction|count|
+-----+----------+-----+
|    1|       0.0|   47|
|    0|       0.0|  987|
|    1|       1.0|  124|
|    0|       1.0|    3|
+-----+----------+-----+

Summary Stats
accuracy = 0.957
weighted precision = 0.958
weighted recall = 0.957
areaUnderROC = 0.985
'''
