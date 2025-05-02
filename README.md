<h1 align="center">
    ML-Microservice
</h1>

## About

This repository is about the **"ML Project Lifecycle"** and my ability to create dependable, deployable machine learning microservice applications which can be used for many things: MVPs, APIs, and more.

Because this project is more focused on ML Engineering as opposed to traditional model design, we will be setting up and working with a very simple model for this project.

If you are interested in some of my work in computer vision, model fine-tuning, parameter evaluation, data engineering pipelines, and more, you can check out my current work projects:
* [OCR Service for Engineering Document Indexing and Tagging](https://github.com/dbchristenson/bab-aat)
* [Using LLMs in Geological Field Research with Agents and MCP](https://github.com/dbchristenson/aei-intelligence)

## Methodology

### Data Collection and Sources
* Data Retrieval
  - I will be using a simple apartment rent price dataset based on listings in the Netherlands.

### Data Preparation
* **Data Cleaning**
    - Using Pandas, I load and clean the dataset by tidying up column names.
* **EDA**
    - Exploring the dataset involved several EDA techniques that are personal to the data scientist.
        - Missing values
        - Data types
        - Correlation Coefs
* **Data Augmentation**
    - Encoded the categorical variables from object types to booleans.
    - Imputed missing data or created a label to signify the data was missing, potentially unearthing a pattern or trend between missing data and rent price.
* **Feature Engineering**
    - Specifically, took the 'facilities' category which contained a string of standard facilities in the apartment such as garage, bathroom, etc and split and found all unique 'facility values' and then binarized them for each row.
* **Feature Selection**
    - To avoid overfitting and multicollinearity, we can simply test the data by training a model with it and seeing the score. I tried to take away as many of the more 'complex' features as possible without letting the score drop too far to make the model lightweight and more general.
    - Selecting only the most important features also means that the model is more flexible for the user—perhaps there exists a user who is missing data necessary for a complex model to give a prediction.

### Model Development
* **Algorithm Choice**
    - For simple regression problems such as this one. I tend to choose a tree based regressor like Random Forest Regressor or XGBoost.
    - Although linear regression models would work, I almost always find that tree-based models perform better at a negligible cost to size.
* **Model Training**
    - After preparing the feature matrix and target vector, split the dataset into 80% training and 20% test subsets.
    - Invoke the ```train_model``` helper to fit a RandomForestRegressor on the training data, producing a model ready for evaluation.
* **Model Evaluation**
    - Use the ```evaluate_model``` function to score the fitted model on the held-out test set by computing its R², providing a clear measure of how well the model explains rent-price variance on unseen data.
* **Hyperparameter Tuning**
    - Within ```train_model```, run a GridSearchCV over a grid of n_estimators ([100, 200, 300]) and max_depth ([3, 6, 9, 12, 15]) using 5-fold cross-validation and R² scoring.
    - Automatically select and refit the best hyperparameter combination to maximize predictive performance on the training folds.

### Model Deployment
* **Package Model as App**
    - Structure the code into four modular components:
        - ```collection.py``` for CSV data ingestion
        - ```preparation.py``` for all feature-engineering logic
        - ```model.py``` to orchestrate training, tuning, evaluation, and serialization,
        - ```model_service.py``` which defines a ModelService class that on startup loads (or trains) the pickled model and exposes a predict method.
    - Wrap ```ModelService.predict``` in a lightweight REST API (e.g., Flask or FastAPI), routing JSON feature inputs to live price predictions.
    - Externalize paths and model names via a shared config module, validate incoming payloads against the expected feature schema, and containerize the application for seamless deployment, enabling users or downstream services to retrieve rental estimates in real time.
* **App to Microservice**
* **Deployment**
