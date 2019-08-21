import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import round
from pyspark.ml.feature import StringIndexer, VectorAssembler
import time
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

# Get number of records
print("The data contains %d records." % cars.count(), '\n')

cars = cars.withColumnRenamed("ncyl", "cyl")
cars = cars.withColumn('length_meters', round(cars.length * 0.0254, 3))

cars = cars.withColumn('weight_kg', round(cars.weight / 2.205, 0))

print("Cars with null cyl", cars.filter('cyl IS NULL').count(), '\n')

indexer = StringIndexer(inputCol='type',
                        outputCol='type_idx')
# Assign index values to strings
indexer = indexer.fit(cars)
# Create column with index values
cars = indexer.transform(cars)

#print(cars.toPandas().sample(12))

print(indexer)
# View the first five records
cars.sample(False, .25).show()

# Check column data types
print('\n', cars.dtypes, '\n')

cars.printSchema()

print("\nNull data exists:", pd.isnull(cars), '\n')

assembler = VectorAssembler(inputCols=['cyl', 'weight'], outputCol='features')

# Consolidate predictor columns
cars_assembled = assembler.transform(cars)

# Check the resulting column
cars_assembled.select('cyl', 'weight', 'features').show(5, truncate=False)

spark.stop()
