import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score


url = r"C:\Users\Pulkit\Downloads\winequality-red.csv" 
wine_data = pd.read_csv(url)


wine_data['is_good_quality'] = (wine_data['quality'] >= 7).astype(int)


wine_data.drop('quality', axis=1, inplace=True)


X = wine_data.drop('is_good_quality', axis=1)
y = wine_data['is_good_quality']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)


knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)
knn_predictions = knn_model.predict(X_test)


dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)
dt_predictions = dt_model.predict(X_test)


rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)


def evaluate_model(predictions, y_true):
    acc = accuracy_score(y_true, predictions)
    f1 = f1_score(y_true, predictions)
    print(f'Accuracy: {acc:.4f}')
    print(f'F1 Score: {f1:.4f}')

print("Logistic Regression:")
evaluate_model(lr_predictions, y_test)

print("\nK-Nearest Neighbors:")
evaluate_model(knn_predictions, y_test)

print("\nDecision Trees Classifier:")
evaluate_model(dt_predictions, y_test)

print("\nRandom Forest Classifier:")
evaluate_model(rf_predictions, y_test)
