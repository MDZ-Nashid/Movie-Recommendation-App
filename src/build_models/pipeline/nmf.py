"""
NMF (Non-negative Matrix Factorization) using Surprise library.
Simpler and more effective than NCF for explicit ratings.
"""

from surprise import NMF, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split
import pandas as pd
import pickle
from typing import Dict, List
from collections import defaultdict
from config.model_config import model_settings
import numpy as np


class SurpriseNMF:
    """
    NMF recommendation model using Surprise library.
    
    Args:
        n_factors: Number of latent factors (default: 64)
        n_epochs: Number of training epochs (default: 20)
        biased: Use biases in the model (default: True)
        reg_pu: Regularization term for users (default: 0.06)
        reg_qi: Regularization term for items (default: 0.06)
        reg_bu: Regularization term for user biases (default: 0.02)
        reg_bi: Regularization term for item biases (default: 0.02)
        lr_bu: Learning rate for user biases (default: 0.005)
        lr_bi: Learning rate for item biases (default: 0.005)
        init_mean: Mean of random initialization (default: 0)
        init_std_dev: Standard deviation of random initialization (default: 0.1)
        random_state: Random seed (default: 42)
    """
    
    def __init__(
        self,
        n_factors: int = 15,
        n_epochs: int = 50,
        biased: bool = False,
        reg_pu: float = 0.1,
        reg_qi: float = 0.1,
        random_state: int = 42,
    ):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.biased = biased
        self.reg_pu = reg_pu
        self.reg_qi = reg_qi
        self.random_state = random_state
        
        self.model = None
        self.trainset = None
        self.testset = None
        self.save_path = f'{model_settings.model_path}/{model_settings.model_name_nmf}.pkl'

    def fit(self):
        """
        Train NMF model using preprocessed data.
        """
        from build_models.pipeline.preprocess import preprocess
        
        print("Loading preprocessed data...")
        data = preprocess()
        
        # Create Surprise dataset
        reader = Reader(rating_scale=(0.5, 5.0))
        surprise_data = Dataset.load_from_df(
            data[['userId', 'movieId', 'rating']], 
            reader
        )
        
        # Train-test split (80-20)
        self.trainset, self.testset = train_test_split(
            surprise_data, 
            test_size=0.2, 
            random_state=self.random_state
        )
        
        # Initialize model
        self.model = NMF(
            n_factors=self.n_factors,
            n_epochs=self.n_epochs,
            biased=self.biased,
            reg_pu=self.reg_pu,
            reg_qi=self.reg_qi,
            random_state=self.random_state,
            verbose=True
        )
        
        print("\nStarting Training...")
        print("=" * 80)
        
        # Train model
        self.model.fit(self.trainset)
        
        print("=" * 80)
        print("Training Complete\n")
        
        # Evaluate on test set
        predictions = self.model.test(self.testset)
        rmse = accuracy.rmse(predictions, verbose=True)
        mae = accuracy.mae(predictions, verbose=True)
        
        print(f"\nTest Set Performance:")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
    
    def save_model(self):
        """
        Save trained model to disk in same format as SVD.
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        # Save in dictionary format like SVD
        data = {
            'model': self.model,
            'trainset': self.trainset,
            'testset': self.testset
        }
        
        with open(self.save_path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"\nModel saved at: {self.save_path}")
    
    @staticmethod
    def load_model():
        """
        Load model from disk.
        
        Returns:
            SurpriseNMF instance with loaded model
        """
        from config.model_config import model_settings
        
        path = f"{model_settings.model_path}/{model_settings.model_name_nmf}.pkl"
        
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        model_obj = SurpriseNMF()
        model_obj.model = data['model']
        model_obj.trainset = data.get('trainset')
        model_obj.testset = data.get('testset')
        
        print(f"Model loaded from {path}")
        return model_obj
    
    def evaluate(self, K_values: List[int] = [5, 10, 20]) -> Dict[str, Dict[str, float]]:
        """
        Evaluate model with ranking metrics.
        
        Args:
            K_values: List of K values for top-K evaluation
        
        Returns:
            Dictionary with metrics for each K value
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        if self.testset is None:
            raise ValueError("No test set available. Call fit() first.")
        
        print("\nEvaluating Ranking Metrics...")
        
        # Get predictions for test set
        predictions = self.model.test(self.testset)
        
        # Group predictions by user
        user_predictions = defaultdict(list)
        for uid, iid, true_r, est, _ in predictions:
            user_predictions[uid].append((iid, est, true_r))
        
        # Calculate metrics for each K
        results = {}
        
        for k in K_values:
            precisions = []
            recalls = []
            ndcgs = []
            
            for uid, user_ratings in user_predictions.items():
                # Sort by predicted rating
                user_ratings.sort(key=lambda x: x[1], reverse=True)
                
                # Top-K predictions
                top_k = user_ratings[:k]
                top_k_items = [iid for iid, _, _ in top_k]
                
                # Relevant items (rating >= 3.0)
                relevant_items = [iid for iid, _, true_r in user_ratings if true_r >= 3.0]
                
                if len(relevant_items) == 0:
                    continue
                
                # Precision@K
                relevant_in_top_k = len([iid for iid in top_k_items if iid in relevant_items])
                precision = relevant_in_top_k / k
                precisions.append(precision)
                
                # Recall@K
                recall = relevant_in_top_k / len(relevant_items)
                recalls.append(recall)
                
                # NDCG@K
                dcg = sum([1.0 / np.log2(idx + 2) if top_k[idx][0] in relevant_items else 0.0 
                          for idx in range(len(top_k))])
                idcg = sum([1.0 / np.log2(idx + 2) 
                           for idx in range(min(len(relevant_items), k))])
                ndcg = dcg / idcg if idcg > 0 else 0.0
                ndcgs.append(ndcg)
            
            results[f"Top-{k}"] = {
                'Precision': np.mean(precisions) if precisions else 0.0,
                'Recall': np.mean(recalls) if recalls else 0.0,
                'NDCG': np.mean(ndcgs) if ndcgs else 0.0
            }
        
        return results
    
    def recommend_movies(self, user_id: int, N: int = 10, movies_csv: str = f'{model_settings.movie_name}') -> List[Dict]:
        if self.model is None:
            raise ValueError("Model not loaded.")
        
        try:
            inner_user_id = self.trainset.to_inner_uid(user_id)
        except ValueError:
            return []
        
        movies_df = pd.read_csv(movies_csv)
        
        user_factors = self.model.pu[inner_user_id]
        all_item_factors = self.model.qi
        
        # Simple dot product (no biases since biased=False)
        scores = np.dot(all_item_factors, user_factors)
        
        # NMF scores are unbounded, normalize to rating scale
        # Map [min_score, max_score] → [0.5, 5.0]
        min_score = scores.min()
        max_score = scores.max()
        scores_normalized = 0.5 + (scores - min_score) / (max_score - min_score) * 4.5
        
        raw_item_ids = []
        valid_scores = []
        
        for inner_iid in range(len(scores_normalized)):
            try:
                raw_iid = self.trainset.to_raw_iid(inner_iid)
                raw_item_ids.append(raw_iid)
                valid_scores.append(scores_normalized[inner_iid])
            except ValueError:
                continue
        
        sorted_indices = np.argsort(valid_scores)[::-1]
        top_n_indices = sorted_indices[:N]
        
        recommendations = []
        for idx in top_n_indices:
            movie_id = raw_item_ids[idx]
            score = valid_scores[idx]
            
            movie_info = movies_df[movies_df['movieId'] == movie_id]
            if movie_info.empty:
                continue
            
            movie_info = movie_info.iloc[0]
            recommendations.append({
                'movieId': int(movie_id),
                'title': movie_info['title'],
                'genres': movie_info['genres'],
                'score': float(score)
            })
        
        return recommendations


# Alias for backward compatibility
NCF = SurpriseNMF
