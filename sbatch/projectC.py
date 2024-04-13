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


# # Purpose: This is for checking the validation error and testing error

# In[ ]:


ss=SparkSession.builder.appName("ALS-based Recommendation Systems").getOrCreate()


# In[ ]:


ss.sparkContext.setCheckpointDir("~/scratch")


# In[ ]:


rating_schema = StructType([ StructField("app_id", IntegerType(), True),                             StructField("helpful", IntegerType(), True ),                             StructField("funny", IntegerType(), True ),                             StructField("date", DateType(), True ),                             StructField("is_recommended", BooleanType(), True ),                             StructField("hours", FloatType(), True ),                             StructField("user_id", IntegerType(), True ),                             StructField("review_id", IntegerType(), True ),                            ])


# In[ ]:


DF = ss.read.csv("/storage/home/zql5426/work/Project/recommendations.csv", schema=rating_schema, header=True, inferSchema=False)
DF = DF.select("app_id", "is_recommended", "hours", "user_id")


# In[ ]:


DF.printSchema()


# ## Calculate the average hours of each game

# In[ ]:


avg_hours_df = DF.groupBy("app_id").agg(avg("hours").alias("avg_hours"))
df_with_avghr = DF.join(avg_hours_df, "app_id")


# ## Rating according to is_recommended and average hours

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


# ## Split info_RDD into three groups: 60% training, 20% validation, and 20% testing.

# In[ ]:


training_RDD, validation_RDD, test_RDD = info_RDD.randomSplit([6, 2, 2])


# ## Prepare input (UserID, MovieID) for training, validation and for testing data

# In[ ]:


training_input_RDD = training_RDD.map(lambda x: (x[0], x[1]) )
validation_input_RDD = validation_RDD.map(lambda x: (x[0], x[1]) ) 
testing_input_RDD = test_RDD.map(lambda x: (x[0], x[1]) )


# ## Generate ALS model

# In[ ]:


model = ALS.train(training_RDD, 4, seed=37, iterations=30, lambda_=0.3)


# ## Compute the training error

# In[ ]:


training_prediction_RDD = model.predictAll(training_input_RDD)


# In[ ]:


training_target_output_RDD = training_RDD.map(lambda x: ( (x['user_id'], x['app_id']), x['rating_points'] ) )


# In[ ]:


training_prediction2_RDD = training_prediction_RDD.map(lambda x:((x[0], x[1]), x[2]))


# In[ ]:


training_evaluation_RDD = training_target_output_RDD.join(training_prediction2_RDD)


# In[ ]:


training_error = math.sqrt(training_evaluation_RDD.map(lambda z: (z[1][0] - z[1][1])**2).mean())


# In[ ]:


print('The training error is: ',training_error)


# ## Compute the RMS validation error

# In[ ]:


validation_prediction_RDD = model.predictAll(validation_input_RDD).map(lambda x: ( (x[0], x[1]), x[2] ) )


# In[ ]:


validation_evaluation_RDD = validation_RDD.map(lambda y: ( (y[0], y[1]), y[2] ) ).join(validation_prediction_RDD)


# In[ ]:


validation_error = math.sqrt(validation_evaluation_RDD.map(lambda z: (z[1][0]- z[1][1])**2).mean())


# In[ ]:


print('The RMS validation error is: ',validation_error)


# ## Evaluate the Hyper-parameters with testing data

# In[ ]:


best_k=4
best_iterations=30
best_regularization=0.3


# In[ ]:


seed = 37
model = ALS.train(training_RDD, best_k, seed=seed, iterations=best_iterations, lambda_= best_regularization)
testing_prediction_RDD=model.predictAll(testing_input_RDD).map(lambda x: ((x[0], x[1]), x[2]))
testing_evaluation_RDD= test_RDD.map(lambda x: ((x[0], x[1]), x[2])).join(testing_prediction_RDD)
testing_error = math.sqrt(testing_evaluation_RDD.map(lambda x: (x[1][0]-x[1][1])**2).mean())
print('The Testing Error for rank k =', best_k, ' regularization = ', best_regularization, ', iterations = ',       best_iterations, ' is : ', testing_error)


# In[ ]:


ss.stop()


# In[ ]:




