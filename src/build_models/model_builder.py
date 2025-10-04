from build_models.pipeline.surprise_item_cf import SurpriseItemCF
from build_models.pipeline.surprise_svd import SurpriseSVD
from build_models.pipeline.nmf import SurpriseNMF
from config.model_config import model_settings


class ModelBuilderService:

    def __init__(self):
        self.model_name = model_settings.model_name
        self.model_path = model_settings.model_path

    def train_model(self):
        model = SurpriseItemCF(k=50, min_k=5)
        model.fit()
        model.save_model()
        results = model.evaluate(K_values=[5, 10, 20])

        print("\n" + "="*60)
        print("Surprise Item-Based CF - Evaluation Results")
        print("="*60)

        for k, metrics in results.items():
            print(f"\n{k}:")
            for metric_name, value in metrics.items():
                print(f"  {metric_name}: {value:.4f}")

    def train_model_svd(self):
        """
        Train Surprise SVD recommendation model.
        """
        model = SurpriseSVD(
            n_factors=150,
            n_epochs=40,
            lr_all=0.002,
            reg_all=0.05,
            biased=True,
            random_state=42
        )

        model.fit()
        model.save_model()
        results = model.evaluate(K_values=[5, 10, 20])

        print("\n" + "="*60)
        print("Surprise SVD - Evaluation Results")
        print("="*60)

        for k, metrics in results.items():
            print(f"\n{k}:")
            for metric_name, value in metrics.items():
                print(f"  {metric_name}: {value:.4f}")

    def train_model_nmf(self):
        """
        Train NMF (Non-negative Matrix Factorization) model.
        """
        model = SurpriseNMF()

        model.fit()
        model.save_model()
        results = model.evaluate(K_values=[5, 10, 20])

        print("\n" + "="*60)
        print("Surprise NMF - Evaluation Results")
        print("="*60)

        for k, metrics in results.items():
            print(f"\n{k}:")
            for metric_name, value in metrics.items():
                print(f"  {metric_name}: {value:.4f}")
