import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import regexp_replace
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
import pandas as pd

# Creating a SparkSession
spark = SparkSession.builder \
                    .master('local[*]') \
                    .appName('first_spark_application') \
                    .getOrCreate()

books = spark.read.csv('/Users/wel51x/Box Sync/MyBox/Code/DataCamp/data/books.csv',
                         sep='|',
                         header=True,
                         inferSchema=True,
                         nullValue='NA')

# Get number of records
print("The data contain %d records." % books.count(), '\n')

# Check column data types
print('\n', books.dtypes, '\n')

books.printSchema()

# Regular expression (REGEX) to match commas and hyphens
REGEX = '[,\\-]'
books = books.withColumn('text', regexp_replace(books.title, REGEX, ' '))

# Text to tokens
books = Tokenizer(inputCol="text", outputCol="tokens").transform(books)

# Take a look at the list of stop words
stopwords = StopWordsRemover()
stopwords.getStopWords()

# Specify the input and output column names
stopwords = stopwords.setInputCol('tokens').setOutputCol('words')
books = stopwords.transform(books)

# Feature hashing
hasher = HashingTF(inputCol="words", outputCol="hash", numFeatures=32)
books = hasher.transform(books)

# Dealing with common words
books = IDF(inputCol="hash", outputCol="features").fit(books).transform(books)

# View the first five records
books.show(truncate=False)

spark.stop()
