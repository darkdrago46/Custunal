try:
    import googleclouddebugger
    googleclouddebugger.enable()
except ImportError:
    pass

import pandas as pd

df = pd.read_csv('C:/Users/VIHITHA KOTA/PycharmProjects/custanal/sent.csv')

df.polarity.value_counts()

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=1000)
vectors = vectorizer.fit_transform(df.text)
words_df = pd.DataFrame(vectors.toarray(), columns=vectorizer.get_feature_names())
words_df.head()

X = words_df
y = df.polarity

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(C=1e9, solver='lbfgs', max_iter=1000)
logreg.fit(X, y)

pd.set_option("display.max_colwidth", 200)

unknown = pd.DataFrame({'content': [
    "I love love love love this kitten",
    "I hate hate hate hate this keyboard",
    "I'm not sure how I feel about toast",
    "Did you see the baseball game yesterday?",
    "The package was delivered late and the contents were broken",
    "Trashy television shows are some of my favorites",
    "I'm seeing a Kubrick film tomorrow, I hear not so great things about it.",
    "I find chirping birds irritating, but I know I'm not the only one",
]})
unknown

print(vectorizer.get_feature_names())

unknown_vectors = vectorizer.transform(unknown.content)
unknown_words_df = pd.DataFrame(unknown_vectors.toarray(), columns=vectorizer.get_feature_names())
unknown_words_df.head()

unknown_words_df.shape

unknown['pred_logreg'] = logreg.predict(unknown_words_df)
unknown['pred_logreg_proba'] = logreg.predict_proba(unknown_words_df)[:,1]

unknown

df.head()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y)

logreg.fit(X_train, y_train)

y_true = y_test
y_pred = logreg.predict(X_test)

from sklearn.metrics import accuracy_score
print("Accuracy score = ",accuracy_score(y_true,y_pred)*100)

y_true
print("P1!")

from flask import Flask, render_template, request

app = Flask(__name__,template_folder="C:/Users/VIHITHA KOTA/PycharmProjects/custanal")

@app.route("/")
def home():
    print("P4!")
    return render_template("w1.html")

@app.route("/w2")
def wp2():
    print("P5!")
    return render_template("w2.html")

@app.route("/w3")
def wp3():
    print("P6!")
    rev = request.args.get("rev")
    pd.set_option("display.max_colwidth", 200)
    unknown = pd.DataFrame({'content': [
        rev
    ]})
    unknown_vectors = vectorizer.transform(unknown.content)
    unknown_words_df = pd.DataFrame(unknown_vectors.toarray(), columns=vectorizer.get_feature_names())
    u = logreg.predict(unknown_words_df)
    if(u[0]==0):
        pred = "Negative"
    else:
        pred = "Positive"
    print("P6!")
    return render_template("w3.html", rev = rev, pred = pred)

if __name__ == "__main__":
    app.run()