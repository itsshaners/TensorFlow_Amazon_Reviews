import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_recommenders as tfrs
from datetime import datetime,timedelta

#Importing Dataset
electronics_data = pd.read_csv("ratings_Electronics.csv", dtype={'rating': 'int8'}, names=['userId', 'productId','rating','timestamp'], index_col=None, header=0)

#Info about Dataset
print("Number of Rating: {:,}".format(electronics_data.shape[0]))
print("Columns: {}".format(np.array2string(electronics_data.columns.values)))
print("Number of Users: {:,}".format(len(electronics_data.userId.unique())))
print("Number of Products: {:,}".format(len(electronics_data.productId.unique())))
electronics_data.describe()['rating'].reset_index()

#Num rating per day
data_by_date = electronics_data.copy()
data_by_date.timestamp = pd.to_datetime(electronics_data.timestamp, unit="s")
data_by_date = data_by_date.sort_values(by="timestamp", ascending=False).reset_index(drop=True)
print("Number of Ratings each day:\n", data_by_date.groupby("timestamp")["rating"].count().tail(10).reset_index())

#Sort product by rating
rating_by_product = electronics_data.groupby("productId").agg({"userId":"count","rating":"mean"}).rename(columns={"userId":"Number of Ratings", "rating":"Average Rating"}).reset_index()
cutoff = 50  #products with more than 50 ratings will be counted
top_rated = rating_by_product.loc[rating_by_product["Number of Ratings"]>cutoff].sort_values(by="Average Rating",ascending=False).reset_index(drop=True)


class RankingModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        embedding_dimension = 32
        unique_userIds = electronics_data['userId'].unique().tolist()
        unique_productIds = electronics_data['productId'].unique().tolist()

        self.user_embeddings = tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary=unique_userIds, mask_token=None),
            tf.keras.layers.Embedding(len(unique_userIds) + 1, embedding_dimension)
        ])

        self.product_embeddings = tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary=unique_productIds, mask_token=None),
            tf.keras.layers.Embedding(len(unique_productIds) + 1, embedding_dimension)
        ])

        # Set up a retrieval task and evaluation metrics over the
        # entire dataset of candidates.
        self.ratings = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(1)
        ])

    def call(self, userId, productId):
        user_embeddings = self.user_embeddings(userId)
        product_embeddings = self.product_embeddings(productId)
        return self.ratings(tf.concat([user_embeddings, product_embeddings], axis=1))


#Final model using the TFRS neural network
class amazonModel(tfrs.models.Model):
    def __init__(self):
        super().__init__()
        self.ranking_model: tf.keras.Model = RankingModel()
        self.task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.RootMeanSquaredError()]
        )

    def compute_loss(self, features, training=False):
        rating_predictions = self.ranking_model(features["userId"], features["productId"])
        return self.task(labels=features["rating"], predictions=rating_predictions)

# Filtering Dataset
cutoff_no_rat = 50  # Only count products which received more than or equal to 50
cutoff_year = 2011  # Only count Rating after 2011
recent_data = data_by_date.loc[data_by_date["timestamp"].dt.year > cutoff_year]
print("Number of Rating: {:,}".format(recent_data.shape[0]))
print("Number of Users: {:,}".format(len(recent_data.userId.unique())))
print("Number of Products: {:,}".format(len(recent_data.productId.unique())))
del data_by_date  #Free up memory
recent_prod = recent_data.loc[recent_data.groupby("productId")["rating"].transform('count').ge(cutoff_no_rat)].reset_index(drop=True)
del recent_data  #Free up memory

#Save unique users, products, and rating
userIds = recent_prod.userId.unique()
productIds = recent_prod.productId.unique()
total_ratings = len(recent_prod.index)
#Save the final ratings
ratings = tf.data.Dataset.from_tensor_slices({"userId":tf.cast(recent_prod.userId.values, tf.string), "productId":tf.cast(recent_prod.productId.values,tf.string), "rating":tf.cast(recent_prod.rating.values, tf.int8,)})

#shuffle rating values
tf.random.set_seed(42)
shuffled = ratings.shuffle(100_000, seed=42, reshuffle_each_iteration=False)

#Split the Dataset 80% training 20% testing
train = shuffled.take( int(total_ratings*0.8))
test = shuffled.skip(int(total_ratings*0.8)).take(int(total_ratings*0.2))
unique_productIds = productIds
unique_userIds = userIds

#Run Model
model = amazonModel()
model.compile(optimizer=tf.keras.optimizers.Adagrad( learning_rate=0.1 ))
cached_train = train.shuffle(100_000).batch(8192).cache()
cached_test = test.batch(4096).cache()
model.fit(cached_train, epochs=10)

#Evaluate Model
model.evaluate(cached_test, return_dict=True)
user_rand = userIds[123]
test_rating = {}

for m in test.take(5):
    rating_model = RankingModel()  # Create a new instance of RankingModel
    test_rating[m["productId"].numpy()] = rating_model(tf.convert_to_tensor([user_rand]), tf.convert_to_tensor([m["productId"].numpy()])) # Use .numpy() to get the actual value)

print("Top 5 recommended products for User {}: ".format(user_rand))
for m in sorted(test_rating, key=test_rating.get, reverse=True):
    print(m.decode())
