from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import mlflow

# load the iris dataset
iris = load_iris()
X = iris.data

# create an experiment called iris_experiment
# create a run inside the previous experiment
# log the shape of the dataframe in a dict { length of the dataframe, width } --> shape.json
# cluster the data :  try clustering with k = {2..10} , log the parameter inertia as a parameter .
# select the optimal k , smallest inertia
# rerun the clustering with optimal k
# plot the dist of label

try:
    mlflow.create_experiment(name="iris experiment")
except mlflow.exceptions.MlflowException:
    print("experiment already exists")

# create a run inside the experiment

with mlflow.start_run(experiment_id=mlflow.get_experiment_by_name(name="iris experiment").experiment_id):
    # log the shape of the dataframe in a dict { length of the dataframe, width } --> shape.json
    length, width = X.shape
    mlflow.log_dict(dictionary={"length": length, "width": width},
                    artifact_file="shape.json")
    # cluster the data :  try clustering with k = {2..10} , log the parameter inertia as a parameter .

    optimal_inertia = 1e10
    optimal_k = 0

    for k in range(2,51):
        kmeans = KMeans(k, random_state=0,n_init="auto")
        kmeans.fit(X)
        inertia = kmeans.inertia_

        mlflow.log_metric(key="inertia", value=inertia)

        if inertia < optimal_inertia:
            optimal_inertia = inertia
            optimal_k = k

    mlflow.log_param(key="optimal_k", value=optimal_k)
    optimal_kmeans= KMeans(n_clusters=optimal_k,random_state=0)
    optimal_kmeans.fit(X)

    # get labels

    labels = optimal_kmeans.labels_

    fig, ax = plt.subplots()
    ax.hist(labels)

    mlflow.log_figure(figure=fig, artifact_file="labels_hist.png")

