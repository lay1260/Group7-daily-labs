from pyspark.sql import SparkSession
from pyspark.sql.functions import to_date, dayofweek

spark = SparkSession.builder \
    .appName("CalculateStoreDailyAvgTraffic") \
    .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
    .getOrCreate()

file_path = "sales_data.csv" 
traffic_data = spark.read.csv(
    file_path,
    header=True,
    inferSchema=True
)

store_daily_avg = (
    traffic_data
    # 筛选工作日（周一到周五）
    .filter(dayofweek(to_date("Date", "yyyy/MM/dd")).between(2, 6))
    .filter("Order_Quantity > 0")  # 排除无效订单
    .groupBy("State")  # 按地区分组
    .avg("Order_Quantity")  # 计算日均订单量
    .withColumnRenamed("avg(Order_Quantity)", "avg_order_quantity")
    .orderBy("avg_order_quantity", ascending=False)  # 按日均订单量降序排列
)

print("各地区工作日日均订单量TOP20（地区-日均订单量）：")
store_daily_avg.show(20)  # 显示前20名

spark.stop()  # 停止Spark会话
