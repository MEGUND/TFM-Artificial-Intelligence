import pandas as pd
import numpy as np
from typing import Dict, Text
import joblib
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors  
import tensorflow as tf
import tensorflow_recommenders as tfrs
from sklearn.preprocessing import MinMaxScaler
from darts.models import TFTModel
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler


class HybridRecommender:
    def __init__(self) -> None:
        self.df = pd.read_csv('data/master_data.zip', compression="zip")[["userId", "movieId", "rating"]]
        #Movie names
        self.movie_dict = joblib.load("data/movie_dict.pkl")
        self.movie_to_id_dict = joblib.load("data/movie_to_id.pkl")
        
        # Built user-item matrix
        self.pivot = self.df.pivot(
            index='userId',
            columns='movieId',
            values='rating'
        ).fillna(0)
        
        # Model has to be re-built due to load issues

        unique_movie_ids = joblib.load("data/unique_movie_ids.pkl")
        unique_user_ids = joblib.load("data/unique_user_ids.pkl")

        class ModelRanking(tf.keras.Model):

            def __init__(self):
                super().__init__()
                embedding_dims = 32

                # User embeddings
                self.user_embeddings = tf.keras.Sequential([
                tf.keras.layers.StringLookup(
                    vocabulary=unique_user_ids, mask_token=None),
                tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dims)
                ])

                # Movie Embeddings
                self.movie_embeddings = tf.keras.Sequential([
                tf.keras.layers.StringLookup(
                    vocabulary=unique_movie_ids, mask_token=None),
                tf.keras.layers.Embedding(len(unique_movie_ids) + 1, embedding_dims)
                ])

                # Predictions
                self.ratings = tf.keras.Sequential([
                # multiple dense layers
                tf.keras.layers.Dense(256, activation="relu"),
                tf.keras.layers.Dense(64, activation="relu"),
                # Ratings in output layer
                tf.keras.layers.Dense(1)
            ])

            def call(self, inputs):
                user_id, movie_id = inputs

                user_embed = self.user_embeddings(user_id)
                movie_embed = self.movie_embeddings(movie_id)

                return self.ratings(tf.concat([user_embed, movie_embed], axis=1))
            
        
        class ModelMovielens(tfrs.models.Model):

            def __init__(self):
                super().__init__()
                self.ranking_model: tf.keras.Model = ModelRanking()
                self.task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
                loss = tf.keras.losses.MeanSquaredError(),
                metrics=[tf.keras.metrics.RootMeanSquaredError()]
                )

            def call(self, features: Dict[str, tf.Tensor]) -> tf.Tensor:
                return self.ranking_model(
                    (features["user_id"], features["movie_id"]))

            def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
                labels = features.pop("rating")
                rating_predictions = self(features)

                # Compute loss and metric
                return self.task(labels=labels, predictions=rating_predictions)
            
        
        self.recommendation_model = ModelMovielens()
        # Dummy input to reconstruct the model
        self.recommendation_model({
            "user_id": np.array(["0"]),
            "movie_id": np.array(["0"])
        })

        self.recommendation_model.load_weights('data/recommendation_model_weights.h5')
        
        # Movies to predict
        all_movies = np.array(list(self.movie_dict.keys()))
        excluded_movies = np.array(joblib.load("data/excluded_movie_ids.pkl"))
        exc_mask = np.isin(all_movies, excluded_movies, invert=True)
        self.candidate_movies = all_movies[exc_mask]
        
        #Forecast loads
        self.forecast_model = TFTModel.load("data/forecasting/forecasting-model.pkl")
        self.target_ts_scaled_dict = joblib.load("data/forecasting/target_ts_scaled_dict.pkl")
        self.id_to_order = dict(zip(list(self.target_ts_scaled_dict.keys()), range(len(self.target_ts_scaled_dict.keys()))))
        self.id_to_order = {str(key): value for key,value in self.id_to_order.items()}
        self.order_to_id = {value:key for key,value in self.id_to_order.items()}
        self.covariate_ts_scaled_dict = joblib.load("data/forecasting/covariate_ts_scaled_dict.pkl")
        self.target_scaler = joblib.load("data/forecasting/target_scaler.pkl")
        
                
    def layer1(self, user_ratings: dict):
     
        # USER INPUT
        #Movie ID - Rating
        self.new_user_ratings= {self.movie_to_id_dict.get(k):v for k, v in user_ratings.items()}
        
        self.pivot = self.pivot.append(self.new_user_ratings, ignore_index=True)

        self.pivot.fillna(0, inplace=True)
        df_sparse = csr_matrix(self.pivot.values)

        model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
        model_knn.fit(df_sparse)

        distances, indices = model_knn.kneighbors(self.pivot.tail(1), n_neighbors=2)

        self.most_similar_user = indices[0][1]
        

    def layer2(self):
        watched_movies = np.array(list(self.new_user_ratings.keys()))
        mask = np.isin(self.candidate_movies, watched_movies, invert=True)
        self.candidate_movies = self.candidate_movies[mask]
        
        movies_to_predict = list(self.candidate_movies.astype(str))
        user_to_predict = str(self.most_similar_user)
        
        recommendation_ratings = {}
        for movie_id in movies_to_predict:
        #movie_name = movie_dict.get(int(movie_id))
            predicted_rating = self.recommendation_model({
                "user_id": np.array([user_to_predict]),
                "movie_id": np.array([movie_id])
            }).numpy()[0][0]
            
            recommendation_ratings[movie_id] = predicted_rating    
        
        #Scale the values to be used in Layer 4
        rec_values = np.array(list(recommendation_ratings.values())).reshape(-1, 1)

        recommend_scaler = MinMaxScaler()
        rec_scaled = recommend_scaler.fit_transform(rec_values)

        self.recommendation_ratings = {key: value for key, value in zip(recommendation_ratings.keys(), rec_scaled.flatten())}
        
        #If seasonality is not included
        return [self.movie_dict[int(ID)] for ID in sorted(self.recommendation_ratings, key=self.recommendation_ratings.get, reverse=True)]
    
    def layer3(self, chosen_month:int):
        
        self.target_ts_scaled_dict = {str(key): value for key, value in self.target_ts_scaled_dict.items() if str(key) in self.recommendation_ratings}
        self.covariate_ts_scaled_dict = {str(key): value for key, value in self.covariate_ts_scaled_dict.items() if str(key) in self.recommendation_ratings}
        
        targets = [x[:-12] for x in self.target_ts_scaled_dict.values()]
        covariates = list(self.covariate_ts_scaled_dict.values())
        
        forecast_preds = self.forecast_model.predict(n=12, series=targets, past_covariates=covariates, n_jobs=-1, verbose=0)
        forecast_preds = dict(zip(list(self.target_ts_scaled_dict.keys()), forecast_preds))
        
        temp_ts = [TimeSeries.from_values(np.array([None])) for _ in range(len(self.id_to_order))]

        for i in self.target_ts_scaled_dict.keys():
            order = self.id_to_order[i]
            temp_ts[order] = forecast_preds[i]

        inversed = self.target_scaler.inverse_transform(temp_ts, n_jobs=-1)

        forecast_preds = {}
        for n, i in enumerate(inversed):
            id_n = self.order_to_id[n]
            if id_n in self.target_ts_scaled_dict.keys():
                forecast_preds[id_n] = i

        pred_arrays = [x.values() for x in forecast_preds.values()]
        pred_arrays = np.concatenate(pred_arrays, axis=0).reshape(-1,)
        pred_arrays =TimeSeries.from_values(pred_arrays)

        aggregate_scaler = Scaler()
        aggregate_scaler.fit(pred_arrays)

        #Scale the values of the chosen month to be used in Layer 4
        monthly_preds = {}
        for key, value in forecast_preds.items():
            month_indexer = value.time_index.month.get_loc(chosen_month)
            monthly_preds[key] = value[month_indexer].values()[0][0]
            
        scaled_preds = aggregate_scaler.transform(TimeSeries.from_values(np.array(list(monthly_preds.values()))))
        scaled_preds = scaled_preds.values().reshape(-1,).tolist()

        self.monthly_preds = {key: value for key, value in zip(monthly_preds.keys(), scaled_preds)}
        
    def ensemble_algo (self, normalized_rating, normalized_forecast):
        highest = np.max([normalized_rating, normalized_forecast])
        output = highest + np.mean([normalized_rating, normalized_forecast])
        
        #Maximum possible value for output is 2 and minimum is 0. Values are normalized according to this.
        normalized_ensembled_output = output/2
        return normalized_ensembled_output
    
    def layer4(self, n):
        aggregated = {key: self.ensemble_algo(self.recommendation_ratings[key], self.monthly_preds[key]) for key in self.recommendation_ratings}
        
        sort_aggregated = sorted(aggregated, key=aggregated.get, reverse=True)
        sorted_movie_names = [self.movie_dict[int(ID)] for ID in sort_aggregated]

        return sorted_movie_names[:n]
        


