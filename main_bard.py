import mlflow
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load the iris dataset
iris = load_iris()

# Create a list of test_size values to loop over
test_size_values = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

# Loop over the test_size values
for test_size in test_size_values:

    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=test_size)
    # Create and train a logistic regression model
    clf = RandomForestClassifier(n_estimators=10)
    clf.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = clf.predict(X_test)

    # Calculate the accuracy
    accuracy = clf.score(X_test, y_test)
    with mlflow.start_run(nested=True):

        # Log the metric accuracy
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_param("test_size", test_size)

    # Print the accuracy
    print("Test size:", test_size, "Accuracy:", accuracy)
