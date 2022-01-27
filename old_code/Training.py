import pandas as pd
from sklearn.model_selection import train_test_split
df = pd.read_csv('posecoords.csv')
#df = pd.read_csv('sittingcoords1.csv')
print("df.head(): ")
print(df.head())
print("############################")
print("df.tail(): ")
print(df.tail())
print("############################")
X = df.drop('class', axis=1)
y = df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)
print("y_test: ")
print(y_test)
print("############################")


from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
pipelines = {
    'lr':make_pipeline(StandardScaler(), LogisticRegression()),
    'rc':make_pipeline(StandardScaler(), RidgeClassifier()),
    'rf':make_pipeline(StandardScaler(), RandomForestClassifier()),
    'gb':make_pipeline(StandardScaler(), GradientBoostingClassifier()),
}
fit_models = {}
for algo, pipeline in pipelines.items():
    model = pipeline.fit(X_train, y_train)
    fit_models[algo] = model
print("fit_models:  ")
print(fit_models)
print("############################")
print("fit_models['rc'].predict(X_test):  ")
print(fit_models['rc'].predict(X_test))
print("############################")
from sklearn.metrics import accuracy_score # Accuracy metrics 
import pickle 
for algo, model in fit_models.items():
    yhat = model.predict(X_test)
    print(algo, accuracy_score(y_test, yhat))
print(fit_models['rf'].predict(X_test))
print(y_test)
with open('body_language.pkl', 'wb') as f:
    pickle.dump(fit_models['rf'], f)