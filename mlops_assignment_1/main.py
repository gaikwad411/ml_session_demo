##################################################################################
# MLOPs Assignment 1 - Practical MLflow and model training
#
#
# 1. Install and configure MLflow tracking server
# 2. Create an MLflow experiment and train a simple model using the Iris dataset with Logistic Regression
# 3. Log model parameters, metrics, and the trained model to the MLflow tracking server
#
#
# This script demonstrates a simple MLflow workflow for training, logging, and
# registering a machine learning model using the Iris dataset and a Logistic    
# Regression model.
#
#
# Github Repository link: https://github.com/gaikwad411/ml_session_demo/blob/main/mlops_assignment_1/main.py
##################################################################################

import mlflow
from mlflow.models import infer_signature

import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Load the Iris dataset
print('Loading the iris dataset')
X, y = datasets.load_iris(return_X_y=True)

# Split the data into training and test sets
print('Splitting the data into training and test sets')
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define the model hyperparameters
print('Defining model hyperparameters')
params = {
    "solver": "lbfgs",
    "max_iter": 1000,
    "multi_class": "auto",
    "random_state": 8888,
}

# Train the model
print('Training the Logistic Regression model')
lr = LogisticRegression(**params)
lr.fit(X_train, y_train)

# Predict on the test set
print('Making predictions on the test set')
y_pred = lr.predict(X_test)

# Calculate metrics
print('Calculating model accuracy')
accuracy = accuracy_score(y_test, y_pred)


# Set our tracking server uri for logging
print('Setting MLflow tracking URI')
# Before this You should run the mlfow server using the command: mlflow server --host 127.0.0.1 --port 5000
mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")

# Create a new MLflow Experiment
print('Setting MLflow experiment')
mlflow.set_experiment("MLflow Quickstart")

# Start an MLflow run
print('Starting MLflow run to log model, parameters, and metrics')
with mlflow.start_run():
    # Log the hyperparameters
    print('Logging model parameters and metrics')
    mlflow.log_params(params)

    # Log the loss metric
    print('Logging model accuracy')
    mlflow.log_metric("accuracy", accuracy)

    # Infer the model signature
    print('Inferring model signature')
    signature = infer_signature(X_train, lr.predict(X_train))

    # Log the model, which inherits the parameters and metric
    print('Logging the model')
    model_info = mlflow.sklearn.log_model(
        sk_model=lr,
        name="iris_model",
        signature=signature,
        input_example=X_train,
        registered_model_name="tracking-quickstart",
    )

    # Set a tag that we can use to remind ourselves what this model was for
    print('Setting model tags')
    mlflow.set_logged_model_tags(
        model_info.model_id, {"Training Info": "Basic LR model for iris data"}
    )


# Load the model back for predictions as a generic Python Function model
print('Loading the model back for inference')
loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)

print('Generating predictions')
predictions = loaded_model.predict(X_test)

print('Comparing the predictions against the actual values')
iris_feature_names = datasets.load_iris().feature_names

print('Displaying the results')
result = pd.DataFrame(X_test, columns=iris_feature_names)
result["actual_class"] = y_test
result["predicted_class"] = predictions
print(result[:4])
