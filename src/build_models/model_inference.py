# model_inference_surprise.py
from pathlib import Path
import pickle as pk
from typing import Dict, List
from config.model_config import model_settings
from build_models.model_builder import ModelBuilderService
from build_models.pipeline.surprise_item_cf import SurpriseItemCF
from build_models.pipeline.surprise_svd import SurpriseSVD
from build_models.pipeline.nmf import SurpriseNMF


class ModelInferenceService:

    def __init__(self):
        self.model = None

# load normal cf model for prediction
    def load_model(self):
        """Load Surprise model from disk."""
        path = Path(f"{model_settings.model_path}/"
                    f"{model_settings.model_name}.pkl")

        if not path.exists():
            print("Model is not trained")
            print("-"*80)
            print("creating new model")
            build_model = ModelBuilderService()
            build_model.train_model()

        with open(path, 'rb') as f:
            data = pk.load(f)

        self.model = SurpriseItemCF()
        self.model.model = data['model']
        self.model.trainset = data['trainset']
        self.model.testset = data['testset']
        self.model.movie_info_dict = data['movie_info_dict']

        print(f"Surprise model loaded from {path}")

# load svd model for prediction
    def load_model_svd(self):
        path = Path(f"{model_settings.model_path}/"
                    f"{model_settings.model_name_svd}.pkl")

        if not path.exists():
            print("Model is not trained")
            print("-"*80)
            print("creating new model")
            build_model = ModelBuilderService()
            build_model.train_model_svd()

        self.model = SurpriseSVD.load_model()

# load NCF model for prediction
    def load_model_nmf(self):
        path = Path(f"{model_settings.model_path}/"
                    f"{model_settings.model_name_nmf}.pkl")

        if not path.exists():
            print("Model is not trained")
            print("-"*80)
            print("creating new model")
            build_model = ModelBuilderService()
            build_model.train_model_nmf()  # Update method name

        # Use static method
        self.model = SurpriseNMF.load_model()

# create recommendation
    def recommend_movies(self, user_id: int, N: int) -> List[Dict]:
        """Generate recommendations."""
        if self.model is None:
            raise ValueError("File not found")
        return self.model.recommend_movies(user_id, N)
