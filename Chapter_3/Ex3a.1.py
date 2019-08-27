import pyspark
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, OneHotEncoderEstimator

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

# Remove the 'flight' column
flights = flights.drop('flight')

# Remove records with missing values in any column and get the number of remaining rows
flights = flights.dropna()
print("\nThe data contains %d records after dropping records with na values." % flights.count())

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

flites = flights_indexed.select('carrier', 'org', 'org_idx')

# Create an instance of the one hot encoder
onehot = OneHotEncoderEstimator(inputCols=["org_idx"], outputCols=["org_dummy"])

# Apply the one hot encoder to the flights data
onehot = onehot.fit(flites)
flights_onehot = onehot.transform(flites)

# Check the results
flights_onehot.select('org', 'org_idx', 'org_dummy').distinct().sort('org_idx').show()

spark.stop()
