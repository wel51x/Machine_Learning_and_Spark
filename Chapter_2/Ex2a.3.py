import pyspark
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer

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

# Remove records with missing 'delay' values
flights = flights.filter('delay IS NOT NULL')

# Create an indexer for carrier categorical feature
indexer = StringIndexer(inputCol="carrier", outputCol='carrier_idx')

# Indexer identifies categories in the data
indexer_model = indexer.fit(flights)

# Indexer creates a new column with numeric index values
flights_indexed = indexer_model.transform(flights)

# Repeat the process for the org categorical feature
flights_indexed = StringIndexer(inputCol="org", outputCol='org_idx').fit(flights_indexed).transform(flights_indexed)
# Check first five records
flights_indexed.show(5)

# Get number of records
print("The data contain %d records." % flights_indexed.count(), '\n')

spark.stop()
