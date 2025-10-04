from build_models.pipeline.collection import load_data



def preprocess():

    #dataset loading
    dataset = load_data()

    #drop timestamps
    dataset = dataset.drop('timestamp', axis=1)

    return dataset