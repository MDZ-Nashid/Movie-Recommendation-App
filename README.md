This is a microservice project which recommends movies for existing users.

Model Names:
1. Item Based Collaborative Filtering [Model no : 1 ]
2. SVD(Single Value Decomposition) [Model no : 2 ]
3. NMF(Non-negative Matrix Factorization) [Model no : 3 ]

These models take 2 input
1. userId
2. no of movies

Input Direction: 
    If you want to give input to find out the prediction results for different models, then follow these instructions.
    1. for model 1, go to src/runner_prediction.py
    2. for model 2, go to src/runner_prediction_svd.py
    3. for model 3, go to src/runner_prediction_nmf.py
    4. find user_id = X and insert your desired user id(int).
    5. find no_movies = X and insert your desired no of movies(int).


File Descriptions:

root files:

pyproject.toml comtains the required dependencies for this project and poetry.lock contains exact versions of each dependencies.

clean.py contains the code which will clear out pycache files which are unnecessary and makes project folder messi.

Makefile contains the code for project automation.


src folder:

runner_prediction.py contains the code which will call model 1 and run prediction
runner_prediction_svd.py contains the code which will call model 2 and run prediction
runner_predictio_nmf.py contains the code which will call model 3 and run prediction

runner_builder.py contains the code which will start building model 1 and evaluate it
runner_builder_svd.py contains the code which will start building model 1 and evaluate it
runner_builder_nmf.py contains the code which will start building model 1 and evaluate it

movies.csv contains dataset of movie_id, title and genre
ratings.csv contains dataset of userId, movieId, rating and timestamp


src/config:

model_config.py contains the configuration file of the whole project which is connected with .env file to centralize the whole project and to control important variable name.
.env file contains the variable names and model or dataset path with their names also


src/build_models:

model_builder.py is accountable for the core logic of building model 1/model 2/model 3. If you want to tune hyperparameters, then go to this file and you can find the main classes for each   
                    model and can tune hyper parameter as you want.

model_inference.py is accountable for loading models and also if model is not found then automatically calls for training and building model.


src/build_models/pipeline:

surprise_svd.py contains the core logic of SVD (model 2) which uses surprise library to build the Recommendation System Model.

surprise_item_cf.py contains the core logic of item based collaborative filtering (model 1) which also uses Surprise library to build Recommendation System Model.

nmf.py contains the the core logic of NMF (model 3) and this also uses Surprise library.

preprocess.py contains the code for preprocessing the dataset.

collection.py is accountable for loading dataset.


src/build_models/models:

contains all saved .pkl file which are accountable for predicting new user's movie set without training the model again.


Code Running Instructions:
first your pc needs to have make installed. To install make, you need to install choco also.

# must install poetry. use this command to install poetry in you powershel in windows (run as admin)
    (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -

1. Install choco + make:
    a. Open powershell as admin
    Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
    b. run the upper command to install choco.
    c. verify by using command "choco --version"
    d. choco install make

3. To run model 1:
    i. open the full project movie_recommendation_system in vscode
    ii. make sure you are in root folder like this "C:\Users\X\Desktop\movie_recommendation_system"
    iii. Open Terminal from top navigation bar
    iv. insert command: make

4. To run model 2:
    i. follow first 3 rule from (to run model 1)
    ii. insert command : make svd

5. To run model 3:
    i. follow first 3 rule from (to run model 1)
    ii. insert command : make nmf

6. to just build and see the evaluation result of model 1:
    i. follow first 3 rule from (to run model 1)
    ii. insert command : make build

7. to just build and see the evaluation result of model 2:
    i. follow first 3 rule from (to run model 1)
    ii. insert command : make build_svd

8. to just build and see the evaluation result of model 3:
    i. follow first 3 rule from (to run model 1)
    ii. insert command : make nmf_builder
