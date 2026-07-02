import pandas as pd
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score

data = pd.read_csv("placement.csv")
data

data.info()

data.isna().sum()

data.shape

data.dtypes

data.duplicated().sum()

data.describe()

data["placement"].value_counts().plot(kind="bar")

def count_outliers_iqr(df, thresh=1.5):
    result = []
    for col in df.select_dtypes(include=['number']).columns:
        
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - thresh * IQR
        upper_bound = Q3 + thresh * IQR

        count = (
            (df[col] < lower_bound) |
            (df[col] > upper_bound)
        ).sum()

        result.append([col, count])

    return pd.DataFrame(
        result,
        columns=["Column", "Outlier_Count"]
    )

outlier_report = count_outliers_iqr(data)
outlier_report

X = data[["cgpa", "iq"]]
y = data["placement"]

model = LogisticRegression()
model.fit(X, y)

kfold = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scores = cross_val_score(
    model,
    X,
    y,
    cv=kfold,
    scoring="accuracy"
)

print("Accuracy for each fold:")
for i, score in enumerate(scores, start=1):
    print(f"Fold {i}: {score:.2f}")
print(f"\nMean Accuracy: {scores.mean():.2f}")
print(f"Standard Deviation: {scores.std():.2f}")

with open("model.pkl", "wb") as file:
    pickle.dump(model, file)