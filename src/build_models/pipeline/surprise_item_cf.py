"""
Item-Based CF using Surprise library
"""

import pickle
import pandas as pd
from surprise import KNNBasic, Dataset, Reader
from surprise.model_selection import train_test_split as surprise_split
from typing import Dict, List
from config.model_config import model_settings
from build_models.pipeline.preprocess import preprocess
import numpy as np


class SurpriseItemCF:
    
    def __init__(self, k: int = 50, min_k: int = 5):
        self.k = k
        self.min_k = min_k
        self.model = None
        self.trainset = None
        self.testset = None
        self.movie_info_dict = None
        
    def fit(self):
        print("Loading data...")
        ratings_df = preprocess()
        
        reader = Reader(rating_scale=(0.5, 5.0))
        data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)
        
        print("Splitting data...")
        self.trainset, self.testset = surprise_split(data, test_size=0.2, random_state=42)
        
        print("Training item-based CF...")
        sim_options = {
            'name': 'cosine',
            'user_based': False,
            'min_support': 5
        }
        
        self.model = KNNBasic(k=self.k, min_k=self.min_k, sim_options=sim_options)
        self.model.fit(self.trainset)
        
        movies_df = pd.read_csv(model_settings.movie_name)
        self.movie_info_dict = movies_df.set_index('movieId')[['title', 'genres']].to_dict('index')
        
        print("Training complete")
        
    def recommend_movies(self, user_id: int, N: int) -> List[Dict]:
        """
        Generate recommendations for a user.
        
        Args:
            user_id (int): User ID
            N (int): Number of recommendations
            
        Returns:
            List[Dict]: Recommendations with movieId, score, title, genres
        """
        if self.model is None:
            raise ValueError("Model not trained")
        
        try:
            inner_uid = self.trainset.to_inner_uid(user_id)
        except ValueError:
            print(f"User {user_id} not in trainset. Returning popular items.")
            # Return top-rated items as fallback
            all_items = list(self.trainset.all_items())[:N]
            fallback = []
            for inner_iid in all_items:
                raw_iid = self.trainset.to_raw_iid(inner_iid)
                mid = raw_iid
                fallback.append({
                    'movieId': mid,
                    'score': 4.0,
                    'title': self.movie_info_dict.get(mid, {}).get('title', 'Unknown'),
                    'genres': self.movie_info_dict.get(mid, {}).get('genres', 'Unknown')
                })
            return fallback
        
        # Get user's rated items
        rated_items = set()
        for (iid, rating) in self.trainset.ur[inner_uid]:
            rated_items.add(self.trainset.to_raw_iid(iid))
        
        all_items = [iid for iid in self.trainset.all_items()]
        candidates = [iid for iid in all_items if self.trainset.to_raw_iid(iid) not in rated_items]
        
        predictions = []
        for inner_iid in candidates:
            raw_iid = self.trainset.to_raw_iid(inner_iid)
            pred = self.model.predict(user_id, raw_iid)
            predictions.append({'movieId': raw_iid, 'score': pred.est})
        
        predictions.sort(key=lambda x: x['score'], reverse=True)
        top = predictions[:N]
        
        for item in top:
            mid = item['movieId']
            if mid in self.movie_info_dict:
                item['title'] = self.movie_info_dict[mid]['title']
                item['genres'] = self.movie_info_dict[mid]['genres']
            else:
                item['title'] = 'Unknown'
                item['genres'] = 'Unknown'
                
        return top
    
    def save_model(self):
        """Save model to disk."""
        path = f"{model_settings.model_path}/{model_settings.model_name}.pkl"
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'trainset': self.trainset,
                'testset': self.testset,
                'movie_info_dict': self.movie_info_dict
            }, f)
        print(f"Model saved to {path}")
    
    @staticmethod
    def load_model():
        """Load model from disk."""
        path = f"{model_settings.model_path}/{model_settings.model_name}.pkl"
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        model_obj = SurpriseItemCF()
        model_obj.model = data['model']
        model_obj.trainset = data['trainset']
        model_obj.testset = data['testset']
        model_obj.movie_info_dict = data['movie_info_dict']
        
        print(f"Model loaded from {path}")
        return model_obj

    def evaluate(self, K_values: List[int] = [5, 10, 20]) -> Dict:
        """
        Evaluate model using Precision@K, Recall@K, NDCG@K.
        
        Args:
            K_values (List[int]): K values for evaluation
            
        Returns:
            Dict: Evaluation metrics
        """
        print("\nEvaluating on test set...")
        
        predictions = self.model.test(self.testset)
        
        results = {}
        for K in K_values:
            user_est_true = {}
            for uid, iid, true_r, est, _ in predictions:
                if uid not in user_est_true:
                    user_est_true[uid] = []
                user_est_true[uid].append((est, true_r, iid))
            
            precisions = []
            recalls = []
            ndcgs = []
            
            for uid, ratings in user_est_true.items():
                ratings.sort(key=lambda x: x[0], reverse=True)
                top_k = ratings[:K]
                
                relevant_threshold = 3.5
                relevant_in_top_k = sum(1 for (_, true_r, _) in top_k if true_r >= relevant_threshold)
                total_relevant = sum(1 for (_, true_r, _) in ratings if true_r >= relevant_threshold)
                
                precision = relevant_in_top_k / K if K > 0 else 0
                recall = relevant_in_top_k / total_relevant if total_relevant > 0 else 0
                
                dcg = sum([1.0 / np.log2(idx + 2) if true_r >= relevant_threshold else 0 
                          for idx, (_, true_r, _) in enumerate(top_k)])
                idcg = sum([1.0 / np.log2(idx + 2) 
                           for idx in range(min(K, total_relevant))])
                ndcg = dcg / idcg if idcg > 0 else 0
                
                precisions.append(precision)
                recalls.append(recall)
                ndcgs.append(ndcg)
            
            results[f'K={K}'] = {
                'Precision@K': np.mean(precisions),
                'Recall@K': np.mean(recalls),
                'NDCG@K': np.mean(ndcgs)
            }
        
        return results