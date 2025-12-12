from pyspark.sql import SparkSession
from pyspark.sql.functions import to_date, dayofweek

spark = SparkSession.builder \
    .appName("FindMaxWeekdayStoreTraffic") \
    .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
    .getOrCreate()

file_path = "sales_data.csv"
traffic_data = spark.read.csv(
    file_path,
    header=True,  # 读取表头
    inferSchema=True  # 自动推断字段类型
)

max_traffic_store = (
    traffic_data
    .filter(dayofweek(to_date("Date", "yyyy/MM/dd")).between(2, 6))  # 筛选工作日（周一到周五：2-6）
    .filter("Order_Quantity > 0")  # 排除无效订单
    .groupBy("State")  # 按地区分组
    .sum("Order_Quantity")  # 计算总订单量
    .withColumnRenamed("sum(Order_Quantity)", "total_order_quantity")
    .orderBy("total_order_quantity", ascending=False)  # 按总订单量降序排列
)

print("工作日总订单量最大的地区TOP20（地区-总订单量）：")
max_traffic_store.show(20)  # 显示前20名

spark.stop()  # 停止Spark会话
