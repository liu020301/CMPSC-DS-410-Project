#!/usr/bin/env python
# coding: utf-8

# ### Reference: Some codes from CMPSC/DS 410 Lab6 are used

# In[ ]:


import pyspark
import pandas as pd
import numpy as np
import math


# In[ ]:


from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.types import StructField, StructType, StringType, LongType, IntegerType, FloatType, DateType,BooleanType
from pyspark.sql.functions import col, column, avg, when, udf
from pyspark.sql.functions import expr
from pyspark.sql.functions import split
from pyspark.sql import Row
from pyspark.mllib.recommendation import ALS


# # Purpose: This is for recommendation

# In[ ]:


ss=SparkSession.builder.appName("ALS-based Recommendation Systems").getOrCreate()


# In[ ]:


ss.sparkContext.setCheckpointDir("~/scratch")


# In[ ]:


rating_schema = StructType([ StructField("app_id", IntegerType(), True),                             StructField("helpful", IntegerType(), True ),                             StructField("funny", IntegerType(), True ),                             StructField("date", DateType(), True ),                             StructField("is_recommended", BooleanType(), True ),                             StructField("hours", FloatType(), True ),                             StructField("user_id", IntegerType(), True ),                             StructField("review_id", IntegerType(), True ),                            ])


# ## Read the recommendations.csv file

# In[ ]:


DF = ss.read.csv("/storage/home/zql5426/work/Project/recommendations.csv", schema=rating_schema, header=True, inferSchema=False)
DF = DF.select("app_id", "is_recommended", "hours", "user_id")


# In[ ]:


DF.printSchema()


# ### Calculate the average hours of each game

# In[ ]:


avg_hours_df = DF.groupBy("app_id").agg(avg("hours").alias("avg_hours"))
df_with_avghr = DF.join(avg_hours_df, "app_id")


# ### Rating according to 'is_recommended' and 'average hours'

# In[ ]:


def assign_points(is_recommended, hours, avg_hours):
    if is_recommended and hours >= avg_hours:
        return 4
    elif is_recommended and hours < avg_hours:
        return 3
    elif not is_recommended and hours > avg_hours:
        return 2
    elif not is_recommended and hours < avg_hours:
        return 1


# In[ ]:


assign_points_udf = udf(assign_points, IntegerType())
df_with_points = df_with_avghr.withColumn("rating_point", assign_points_udf(col("is_recommended"), col("hours"), col("avg_hours")))


# In[ ]:


ratings_df = df_with_points.select("user_id", "app_id", "rating_point")
ratings_df.printSchema()


# In[ ]:


ratings_df = ratings_df.withColumn("rating_points", ratings_df["rating_point"].cast("float"))
info_df = ratings_df.select("user_id", "app_id", "rating_points")
info_df.printSchema()
info_df = info_df.na.drop(subset=["rating_points"])


# In[ ]:


info_RDD = info_df.rdd


# In[ ]:


training_input_RDD = info_RDD.map(lambda x: (x[0], x[1]) )


# In[ ]:


model = ALS.train(info_RDD, 4, seed=37, iterations=30, lambda_=0.3)


# In[ ]:


training_prediction_RDD = model.predictAll(training_input_RDD)


# In[ ]:


predict_schema = StructType([ StructField("user_id", IntegerType(), True),                             StructField("app_id", IntegerType(), True ),                             StructField("rating_points", FloatType(), True ),                            ])


# ## Write the prediction ratings to csv files

# In[ ]:


predict_df = spark.createDataFrame(training_prediction_RDD, predict_schema)
output_path = "/storage/home/zql5426/work/Project/Predict_result"
predict_df.write.csv(output_path, mode="overwrite", header=True)


# ## A function for recommending games to a given user_id

# In[ ]:


def recommend_games(user_id):
    games_csv_path = "/storage/home/zql5426/work/Project/games.csv"
    predictions_df = predict_df
    games_df = ss.read.csv(games_csv_path, header=True, inferSchema=True)
    # get the recommended games
    top_predictions = predictions_df.filter(predictions_df.user_id == user_id).orderBy(predictions_df.rating_points.desc()).limit(10)
    
    # get the game names
    recommendations = top_predictions.join(games_df, top_predictions.app_id == games_df.app_id).select(top_predictions.app_id, games_df.title)
    recommended_games = recommendations.collect()
    result = [(row.app_id, row.title) for row in recommended_games]
    
    return result


# ## Change the parameter of 'recommend_games' function below to get the recommended games for your chosen user

# In[ ]:


recommendations = recommend_games(8762579)
for game_id, game_name in recommendations:
    print(f"Game ID: {game_id}, Game Name: {game_name}")


# In[ ]:


ss.stop()


# In[ ]:




