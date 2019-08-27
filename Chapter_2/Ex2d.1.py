import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import regexp_replace
from pyspark.ml.feature import Tokenizer

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
wrangled = Tokenizer(inputCol="text", outputCol="words").transform(wrangled)

# View the first four records
sms.show(4, truncate=False)
wrangled.show(4, truncate=False)

spark.stop()
