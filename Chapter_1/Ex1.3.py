import pyspark
from pyspark.sql import SparkSession
import time
from pyspark.sql.types import StructType, StructField, IntegerType, StringType

# Creating a SparkSession
spark = SparkSession.builder \
                    .master('local[*]') \
                    .appName('first_spark_application') \
                    .getOrCreate()

# Specify column names and types
schema = StructType([
    StructField("id", IntegerType()),
    StructField("text", StringType()),
    StructField("label", IntegerType())
])

sms = spark.read.csv('/Users/wel51x/Box Sync/MyBox/Code/DataCamp/data/SMS.csv',
                         sep=';',
                         header=False,
                         schema=schema)

sms.printSchema()

spark.stop()
