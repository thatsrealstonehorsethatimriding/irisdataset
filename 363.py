from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


iris = load_iris()
X = iris.data  # Features
y = iris.target  # Target variable


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)


knn_classifier = KNeighborsClassifier(n_neighbors=3)  


knn_classifier.fit(X_train, y_train)


y_pred = knn_classifier.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)


print(f"Accuracy: {accuracy}")
print("\nClassification Report:\n", classification_rep)
print("\nConfusion Matrix:\n", conf_matrix)
