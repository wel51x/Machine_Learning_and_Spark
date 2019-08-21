import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import round

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
flights_km = flights.withColumn('km', round(flights.mile * 1.60934, 0)) \
                    .drop('mile')

# Create 'label' column indicating whether flight delayed (1) or not (0)
flights_km = flights_km.withColumn('xdelay', (flights_km.delay >= 15).cast('integer'))

# Check first five records
flights_km.show(5)

# Get number of records
print("The data contain %d records." % flights_km.count(), '\n')

spark.stop()
