#!/usr/bin/env python
# coding: utf-8

# ### Reference: Some codes from CMPSC/DS 410 Lab6 are used

# In[1]:


import pyspark
import pandas as pd
import numpy as np
import math


# In[2]:


from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.types import StructField, StructType, StringType, LongType, IntegerType, FloatType, DateType,BooleanType
from pyspark.sql.functions import col, column, avg, when, udf
from pyspark.sql.functions import expr
from pyspark.sql.functions import split
from pyspark.sql import Row
from pyspark.mllib.recommendation import ALS


# # Purpose: This is for finding the best training parameter with k = [4, 7, 10, 13]

# In[3]:


ss=SparkSession.builder.appName("ALS-based Recommendation Systems").getOrCreate()


# In[ ]:


print(ss.sparkContext.getConf().getAll())


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


# ### Rating according to is_recommended and average hours

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


# ## Split 'info_RDD' into three groups: 60% training, 20% validation, and 20% testing.

# In[ ]:


training_RDD, validation_RDD, test_RDD = info_RDD.randomSplit([3, 1, 1])


# ## Prepare input (UserID, MovieID) for training, validation and for testing data

# In[ ]:


training_input_RDD = training_RDD.map(lambda x: (x[0], x[1]) )
validation_input_RDD = validation_RDD.map(lambda x: (x[0], x[1]) ) 
testing_input_RDD = test_RDD.map(lambda x: (x[0], x[1]) )


# ## Loop through different hyper-parameter settings to find the best

# In[ ]:


## Initialize a Pandas DataFrame to store evaluation results of all combination of hyper-parameter settings
hyperparams_eval_df = pd.DataFrame( columns = ['k', 'regularization', 'iterations', 'validation RMS', 'testing RMS'] )
# initialize index to the hyperparam_eval_df to 0
index =0 
# initialize lowest_error
lowest_validation_error = float('inf')
# Set up the possible hyperparameter values to be evaluated
iterations_list = [15, 30]
regularization_list = [0.1, 0.2, 0.3]
rank_list = [13]
for k in rank_list:
    for regularization in regularization_list:
        for iterations in iterations_list:
            seed = 37
            # Construct a recommendation model using a set of hyper-parameter values and training data
            model = ALS.train(training_RDD, k, seed=seed, iterations=iterations, lambda_=regularization)
            # Evaluate the model using evalution data
            # map the output into ( (userID, movieID), rating ) so that we can join with actual evaluation data
            # using (userID, movieID) as keys.
            validation_prediction_RDD= model.predictAll(validation_input_RDD).map(lambda x: ( (x[0], x[1]), x[2] ) )
            validation_evaluation_RDD = validation_RDD.map(lambda y: ( (y[0], y[1]), y[2] ) ).join(validation_prediction_RDD)
            # Calculate RMS error between the actual rating and predicted rating for (userID, movieID) pairs in validation dataset
            validation_error = math.sqrt(validation_evaluation_RDD.map(lambda z: (z[1][0] - z[1][1])**2).mean())
            # Save the error as a row in a pandas DataFrame
            hyperparams_eval_df.loc[index] = [k, regularization, iterations, validation_error, float('inf')]
            index = index + 1
            # Check whether the current error is the lowest
            if validation_error < lowest_validation_error:
                best_k = k
                best_regularization = regularization
                best_iterations = iterations
                best_index = index - 1
                lowest_validation_error = validation_error
print('The best rank k is ', best_k, ', regularization = ', best_regularization, ', iterations = ',      best_iterations, '. Validation Error =', lowest_validation_error)


# In[4]:


ss.stop()


# In[ ]:




