import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("restaurant_reviews.csv")
df["Sentiment"] = df["Rating"].apply(lambda x: 1 if x >= 3 else 0)

vectorizer = TfidfVectorizer(max_features=1000, stop_words="english")
X = vectorizer.fit_transform(df["Review"]).toarray()
y = df["Sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nModel Performance:")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))
print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))

df["Sentiment_Score"] = model.predict_proba(X)[:, 1]

restaurant_scores = (
    df.groupby("Restaurant")["Sentiment_Score"]
    .mean()
    .sort_values(ascending=False)
    .reset_index()
)

top_n = 5
print(f"\nTop {top_n} Recommended Restaurants (Based on Review Sentiment):\n")
print(restaurant_scores.head(top_n))


def recommend_similar_restaurants(restaurant_name, top_n=3):
    matches = df[df["Restaurant"].str.contains(restaurant_name, case=False, na=False)]
    if matches.empty:
        print("Restaurant not found. Try another name or check spelling.")
        return
    selected_name = matches.iloc[0]["Restaurant"]
    selected_score = restaurant_scores.loc[
        restaurant_scores["Restaurant"] == selected_name, "Sentiment_Score"
    ].values[0]
    recommendations = restaurant_scores[
        (restaurant_scores["Sentiment_Score"] >= selected_score - 0.05)
        & (restaurant_scores["Restaurant"] != selected_name)
    ].head(top_n)
    print(f"\nRestaurants similar to '{selected_name}':\n")
    print(recommendations)
