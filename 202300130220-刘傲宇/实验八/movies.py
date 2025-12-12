# 解析电影种类
movies = movies.withColumn('genre', explode(split(col('genres'), '[|]')))

# 将数据连接起来
ratings_with_genres = ratings.join(movies, on='movieId').join(users, on='userId')

# 各类职业对不同种类电影的关注度
occupation_genre_counts = ratings_with_genres.groupBy('occupation', 'genre').count().toPandas()
occupation_genre_counts['occupation'] = occupation_genre_counts['occupation'].map(occupation_map)

# 生成透视表
heatmap_data = occupation_genre_counts.pivot_table(index='occupation', columns='genre', values='count', fill_value=0)

# 绘制热力图
plt.figure(figsize=(14, 10))
sns.heatmap(heatmap_data, annot=True, fmt='g', cmap='viridis')
plt.title('Occupation vs Genre Heatmap')
plt.xlabel('Genre')
plt.ylabel('Occupation')
plt.xticks(rotation=90)
plt.savefig('occupation_genre_heatmap.png')
plt.show()