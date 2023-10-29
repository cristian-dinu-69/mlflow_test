import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import mlflow

# load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

print(X.shape, y.shape)

# create an experiment called iris_experiment
# create a run inside the previous experiment
# log the shape of the dataframe in a dict { length of the dataframe, width } --> shape.json
# cluster the data :  try clustering with k = {2..10} , log the parameter inertia as a parameter .
# select the optimal k , smallest inertia
# rerun the clustering with optimal k
# plot the dist of label

try:
    mlflow.create_experiment(name="iris classifier")
except mlflow.exceptions.MlflowException:
    print("experiment already exists")

# create a run inside the experiment

with mlflow.start_run(experiment_id=mlflow.get_experiment_by_name(name="iris classifier").experiment_id):
    # log the shape of the dataframe in a dict { length of the dataframe, width } --> shape.json
    length, width = X.shape
    mlflow.log_dict(dictionary={"length": length, "width": width},
                    artifact_file="shape.json")
    # cluster the data :  try clustering with k = {2..10} , log the parameter inertia as a parameter .

    train_list = []
    acc_list = []

    for k in list(np.linspace(start=0.2, stop=0.9, num=8)):
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=k)
        train_lines = int((1-k) * X.shape[0])
        clf = RandomForestClassifier(n_estimators=5,max_depth=2)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        train_list.append(train_lines)
        acc_list.append(round(acc,3))
        print(X_train.shape[0],round(acc,3))
        mlflow.log_metric(key="accuracy", value=round(acc, 3))
        mlflow.log_param(key="train_lines", value=train_lines)
    print("train_list",train_list)
    print("acc_list",acc_list)
    df_acc = pd.DataFrame({"train_list": train_list, "acc":acc_list})
    print(df_acc)
    #     # with mlflow.start_run(nested=True):
    #     # mlflow.log_param(key="train_lines", value=train_lines)
    #     #
    #     # with mlflow.start_run(nested=True):
    #     #     mlflow.log_param(key="train_percent", value=k)
    #     #
    #     # with mlflow.start_run(nested=True):
    #
    #
    fig, ax = plt.subplots()
    ax.plot(df_acc["train_list"], df_acc["acc"],c ="g")

    mlflow.log_figure(figure=fig, artifact_file="accuracy_plot.png")

# Incorrect Example:
# ---------------------------------------exit
# with mlflow.start_run():
#     mlflow.log_param("depth", 3)
#     mlflow.log_param("depth", 5)
# ---------------------------------------
#
# Which will throw an MlflowException for overwriting a
# logged parameter.
#
# Correct Example:
# ---------------------------------------
# with mlflow.start_run():
#     with mlflow.start_run(nested=True):
#         mlflow.log_param("depth", 3)
#     with mlflow.start_run(nested=True):
#         mlflow.log_param("depth", 5)