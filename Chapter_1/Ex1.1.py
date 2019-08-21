import pyspark
from pyspark.sql import SparkSession
import time

# Creating a SparkSession
spark = SparkSession.builder \
                    .master('local[*]') \
                    .appName('first_spark_application') \
                    .getOrCreate()

time.sleep(2)

spark.stop()
