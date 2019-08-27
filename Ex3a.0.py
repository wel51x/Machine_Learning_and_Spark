from pyspark.sql import SparkSession
from pyspark.sql.functions import round
from pyspark.ml.feature import StringIndexer, OneHotEncoderEstimator
from pyspark.mllib.linalg import DenseVector, SparseVector
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

indexer = StringIndexer(inputCol='type',
                        outputCol='type_idx')

# Assign index values to strings
indexer = indexer.fit(cars)
# Create column with index values
cars = indexer.transform(cars)

pd.set_option('display.max_columns', None) # all cols
pd.set_option('display.width', 161)
#print(cars.toPandas().sample(12))

# Check column data types
print('\n', cars.dtypes, '\n')

kars = cars.select('name', 'type', 'type_idx')

print(kars.toPandas().sample(12))

onehot = OneHotEncoderEstimator(inputCols=['type_idx'], outputCols=['type_dummy'])
onehot = onehot.fit(kars)
kars = onehot.transform(kars)
kars.select('type', 'type_idx', 'type_dummy').distinct().sort('type_idx').show()

print("DenseVector:", DenseVector([1, 0, 0, 0, 0, 7, 0, 0]))
print("SparseVector:", SparseVector(8, {0: 1.0, 5: 7.0}))

spark.stop()
