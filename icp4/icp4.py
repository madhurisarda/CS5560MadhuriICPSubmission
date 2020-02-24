# Import library
import pyspark
from pyspark.sql import SQLContext
from pyspark import SparkFiles

# Setting pyspark to variable to leverage its functionality
sc = pyspark.SparkContext()
sqlContext = SQLContext(sc)

# Reading data file in data frame in the form of RDD (Resilient Distributed Dataset)
df = sqlContext.read.csv(SparkFiles.get("D:/Masters/KDM/data.csv"), header=True, inferSchema= True)

# RDD Actions
# Collect all the information present in the data set
action_1 = df.collect()
print("Action 1: Collecting all the information present within the data set")
print(action_1)
# Count number of elements in the data set
action_2 = df.count()
print("Action 2: Count the number of data points present within the data set")
print(action_2)
# Return first 'n' number of elements from the data set
action_3 = df.take(2)
print("Action 3: Taking out 'n' number of data points from the entire data set")
print(action_3)


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