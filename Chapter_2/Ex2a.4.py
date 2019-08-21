import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import round
from pyspark.ml.feature import StringIndexer, VectorAssembler

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

# Convert 'mile' to 'km' and drop 'mile' column
flights = flights.withColumn('km', round(flights.mile * 1.60934, 0)) \
                    .drop('mile')

# Create an indexer for carrier categorical feature
indexer = StringIndexer(inputCol="carrier", outputCol='carrier_idx')

# Indexer identifies categories in the data
indexer_model = indexer.fit(flights)

# Indexer creates a new column with numeric index values
flights_indexed = indexer_model.transform(flights)

# Repeat the process for the org categorical feature
flights_indexed = StringIndexer(inputCol="org", outputCol='org_idx').fit(flights_indexed).transform(flights_indexed)

# Create an assembler object
assembler = VectorAssembler(inputCols=["mon", "dom", "dow", "carrier_idx", "org_idx", "km", "depart", "duration"],
                            outputCol='features')

# Consolidate predictor columns
flights_assembled = assembler.transform(flights_indexed)

# Check the resulting column
flights_assembled.select('features', 'delay').show(5, truncate=False)

# Get number of records
print("The data contain %d records." % flights_indexed.count(), '\n')

spark.stop()
