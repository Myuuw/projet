import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso
from lightgbm.sklearn import LGBMRegressor
from sklearn.cluster import AgglomerativeClustering
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from time import process_time

data = pd.read_csv("creditcard.csv")
X = data.drop(['Class'], axis=1)
Y = data['Class']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Modèle Ridge
r_model = Ridge(alpha=1.0)
start = process_time()
r_model.fit(X_train, y_train)
end = process_time()
start_predict = process_time()
y_pred_ridge = r_model.predict(X_test)
end_predict = process_time()
y_pred_ridge_classes = [1 if pred > 0.5 else 0 for pred in y_pred_ridge]
accuracy_ridge = accuracy_score(y_test, y_pred_ridge_classes)
print("Accuracy Score (Ridge):", accuracy_ridge)
print(end - start)
print(end_predict - start_predict)

# Modèle RandomForestClassifier
model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
start2 = process_time()
model_rf.fit(X_train, y_train)
end2 = process_time()
start_predict2 = process_time()
y_pred_rf = model_rf.predict(X_test)
end_predict2 = process_time()
y_pred_rf_classes = [1 if pred > 0.5 else 0 for pred in y_pred_rf]
accuracy_rf = accuracy_score(y_test, y_pred_rf_classes)
print(end2 - start2)
print(end_predict2 - start_predict2)
print("Accuracy Score (RandomForestClassifier):", accuracy_rf)

# Modèle GradientBoostingClassifier
model_gb = GradientBoostingClassifier(random_state=40)
start3 = process_time()
model_gb.fit(X_train, y_train)
end3 = process_time()
start_predict3 = process_time()
y_pred_gb = model_gb.predict(X_test)
end_predict3 = process_time()
y_pred_gb_classes = [1 if pred > 0.5 else 0 for pred in y_pred_gb]
accuracy_gb = accuracy_score(y_test, y_pred_gb_classes)
print("Accuracy Score (GradientBoostingClassifier):", accuracy_gb)
print(end3 - start3)
print(end_predict3 - start_predict3)

# Modèle LogisticRegression
model = LogisticRegression(random_state=42)
start4 = process_time()
model.fit(X_train, y_train)
end4 = process_time()
start_predict4 = process_time()
y_pred = model.predict(X_test)
end_predict4 = process_time()
y_pred_classes = [1 if pred > 0.5 else 0 for pred in y_pred]
accuracy_lr = accuracy_score(y_test, y_pred_classes)
print("Accuracy Score (LogisticRegression):", accuracy_lr)
print(end4 - start4)
print(end_predict4 - start_predict4)

# Modèle GradientBoostingRegressor
gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=1.0,max_depth=3,random_state=42 )
start5 = process_time()
gb_model.fit(X_train, y_train)
end5 = process_time()
start_predict5 = process_time()
y_pred_gb_model = gb_model.predict(X_test)
end_predict5 = process_time()
y_pred_gb_classes = [1 if pred > 0.5 else 0 for pred in y_pred_gb_model]
accuracy_GradientBoostingRegressor = accuracy_score(y_test, y_pred_gb_classes)
print("Accuracy Score (GradientBoostingRegressor):", accuracy_GradientBoostingRegressor)
print(end5 - start5)
print(end_predict5 - start_predict5)

# Modèle LinearRegression
lr_model = LinearRegression()
start6 = process_time()
lr_model.fit(X_train, y_train)
end6 = process_time()
start_predict6 = process_time()
y_pred_LinearRegression = lr_model.predict(X_test)
end_predict6 = process_time()
y_pred_LinearRegression_classes = [1 if pred > 0.5 else 0 for pred in y_pred_LinearRegression]
accuracy_LinearRegression = accuracy_score(y_test, y_pred_LinearRegression_classes)
print("Accuracy Score (LinearRegression):", accuracy_LinearRegression)
print(end6 - start6)
print(end_predict6 - start_predict6)

# Modèle LGBMRegressor
lg_model = LGBMRegressor()
start7 = process_time()
lg_model.fit(X_train, y_train)
end7 = process_time()
start_predict7 = process_time()
y_pred_lg_model = lg_model.predict(X_test)
end_predict7 = process_time()
y_pred_lg_classes = [1 if pred > 0.5 else 0 for pred in y_pred_lg_model]
accuracy_lg = accuracy_score(y_test, y_pred_lg_classes)
print("Accuracy Score (LGBMRegressor):", accuracy_lg)
print(end7 - start7)
print(end_predict7 - start_predict7)

# Modèle Lasso
r_model = Lasso()
start8 = process_time()
r_model.fit(X_train, y_train)
end8 = process_time()
start_predict8 = process_time()
y_pred_lasso = r_model.predict(X_test)
end_predict8 = process_time()
y_pred_Lasso_classes = [1 if pred > 0.5 else 0 for pred in y_pred_lasso]
accuracy_Lasso = accuracy_score(y_test, y_pred_Lasso_classes)
print("Accuracy Score (Lasso):", accuracy_Lasso)
print(end8 - start8)
print(end_predict8 - start_predict8)

# Modèle DecisionTreeClassifier
model_decision_tree = DecisionTreeClassifier(random_state=40)
start9 = process_time()
model_decision_tree.fit(X_train, y_train)
end9 = process_time()
start_predict9 = process_time()
y_pred_decision_tree = model_decision_tree.predict(X_test)
end_predict9 = process_time()
accuracy_decision_tree = accuracy_score(y_test, y_pred_decision_tree)
print("Accuracy Score (Decision Tree):", accuracy_decision_tree)
print(end9 - start9)
print(end_predict9 - start_predict9)


# Modèle GaussianMixture
gmm = GaussianMixture(n_components=2, random_state=42)
start10 = process_time()
gmm.fit(X_train)
end10 = process_time()
start_predict10 = process_time()
y_pred_gmm = gmm.predict(X_test)
end_predict10 = process_time()
print("Accuracy Score (GaussianMixture):", accuracy_score(y_test, y_pred_gmm))
print(end10 - start10)
print(end_predict10 - start_predict10)

# Modèle KMeans
kmeans = KMeans(n_clusters=2, random_state=42)
start11 = process_time()
kmeans.fit(X_train)
end11 = process_time()
start_predict11 = process_time()
y_pred_kmeans = kmeans.predict(X_test)
end_predict11 = process_time()
print("Accuracy Score (Kmeans):", accuracy_score(y_test, y_pred_kmeans))
print(end11 - start11)
print(end_predict11 - start_predict11)

# Modèle AgglomerativeClustering
agg_clustering = AgglomerativeClustering(n_clusters=2)
start12 = process_time()
agg_clustering.fit_predict(X_train)
end12 = process_time()
start_predict12 = process_time()
y_pred_agg = agg_clustering.fit_predict(X_test)
end_predict12 = process_time()
print("Accuracy Score (AgglomerativeClustering):", accuracy_score(y_test, y_pred_agg))
print(end12 - start12)
print(end_predict12 - start_predict12)
