# Import library
import pyspark
from pyspark.sql import SQLContext
from pyspark import SparkFiles

# Setting pyspark to variable to leverage its functionality
sc = pyspark.SparkContext()
sqlContext = SQLContext(sc)

# Reading data file in data frame in the form of RDD (Resilient Distributed Dataset)
df = sqlContext.read.csv(SparkFiles.get("D:/Masters/KDM/data.csv"), header=True, inferSchema= True)



# RDD Transformations with Actions
# Grouping data and counting their total based on the 'Contract' feature
transform_1 = df.groupBy('Contract').count()
print("Transformation 1: Grouping data based on the Contract feature")
print(transform_1.show())
# Ordering data based on the 'Internet Service' feature
transform_2 = df.orderBy('InternetService').take(5)
print("Transformation 2: Ordering data based on the Internet Service type used")
print(transform_2)
# Identifying any distinct features/elements present within the data
transform_3 = df.distinct().show()
print("Transformation 3: Identifying distinct data elements based on the features")
print(transform_3)