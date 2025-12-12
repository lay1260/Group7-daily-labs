from pyspark.sql import SparkSession

# 创建 SparkSession
spark = SparkSession.builder.appName("WordCount").getOrCreate()

# 读取文本文件
text_file = spark.sparkContext.textFile("text.txt")

# 执行 WordCount 操作
word_counts = (text_file
               .flatMap(lambda line: line.split())  # 将每一行按空格拆分成单词
               .map(lambda word: (word, 1))  # 每个单词映射为 (word, 1)
               .reduceByKey(lambda a, b: a + b)  # 对相同的单词进行累加
               )

# 打印结果
for word, count in word_counts.collect():
    print(f"{word}: {count}")

# 关闭 SparkSession
spark.stop()