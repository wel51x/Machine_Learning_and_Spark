import pyspark
from pyspark.sql import SparkSession
import time
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

# View the first five records
flights.show(5)

# Check column data types
print('\n', flights.dtypes, '\n')

flights.printSchema()

print("\nNull data exists:", pd.isnull(flights))

spark.stop()
