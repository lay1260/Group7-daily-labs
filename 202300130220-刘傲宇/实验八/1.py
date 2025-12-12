from pyspark.sql import SparkSession

# 初始化 SparkSession
spark = SparkSession.builder.appName("SimpleSparkApp").getOrCreate()

# 加载数据
data = spark.read.csv('sales_data.csv',header=True,inferSchema=True)

# 执行一些数据处理操作
result = data.filter(data["Product_Category"] == "Accessories").groupBy("Date").sum("Revenue")

# 显示结果
result.show()