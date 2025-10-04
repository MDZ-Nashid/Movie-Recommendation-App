from build_models.model_inference import ModelInferenceService


def main():
    mdl_svc = ModelInferenceService()
    mdl_svc.load_model()
    
    user_id = 5
    no_movies = 10
    print(f"Testing recommendations for User {user_id}:\n")
    
    recommend = mdl_svc.recommend_movies(user_id, no_movies)
    
    print(f"Top-{no_movies} recommendations:")
    print("-" * 80)
    
    for idx, rec in enumerate(recommend, 1):
        score = rec['score']
        title = rec['title'][:45].ljust(45)
        print(f"{idx:2d}. Score: {score:.4f} - {title} - {rec.get('genres', 'N/A')}")


if __name__ == "__main__":
    main()