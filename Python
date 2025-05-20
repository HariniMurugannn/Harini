from flask import Flask, redirect, url_for, session, request, render_template_string
from flask_dance.contrib.github import make_github_blueprint, github
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np

app = Flask(__name__)
app.secret_key = "supersecretkey"

# GitHub OAuth setup
github_bp = make_github_blueprint(client_id="GITHUB_CLIENT_ID", client_secret="GITHUB_CLIENT_SECRET")
app.register_blueprint(github_bp, url_prefix="/login")

# Simple dummy model training
df = pd.DataFrame({
    'age': [25, 35, 45, 55],
    'bmi': [22, 30, 28, 35],
    'smoker': [0, 1, 0, 1],
    'disease': [0, 1, 0, 1]
})
X = df[['age', 'bmi', 'smoker']]
y = df['disease']
model = LogisticRegression().fit(X, y)

# HTML template for user input
form_template = """
{% if not github.authorized %}
    <a href="{{ url_for('github.login') }}">Login with GitHub</a>
{% else %}
    <form action="/" method="post">
        Age: <input type="number" name="age"><br>
        BMI: <input type="number" name="bmi"><br>
        Smoker (0 or 1): <input type="number" name="smoker"><br>
        <input type="submit" value="Predict">
    </form>
    {% if prediction is not none %}
        <p>Prediction: {{ 'Disease Detected' if prediction else 'No Disease' }}</p>
    {% endif %}
{% endif %}
"""

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if github.authorized:
        if request.method == "POST":
            age = float(request.form["age"])
            bmi = float(request.form["bmi"])
            smoker = int(request.form["smoker"])
            data = np.array([[age, bmi, smoker]])
            prediction = model.predict(data)[0]
    return render_template_string(form_template, prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
