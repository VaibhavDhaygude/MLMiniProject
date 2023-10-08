from flask import Flask, render_template, request

# Importing necessary libraries
import pickle
import joblib

app = Flask(__name__)

# Load the K-means model and scaler
with open('model.pkl', 'rb') as model_file:
    kmeans_model = pickle.load(model_file)

scaler_filename = "scaler.save"
scaler = joblib.load(scaler_filename) 

# Mapping of cluster numbers to descriptions and recommendations
cluster_info = {
    0: "Low Income, High Spending Score (Cluster 1) - May or may not target these group of customers based on the policy of the mall.",
    1: "Average Income, Average Spending Score (Cluster 2) - Can target these set of customers by providing them with Low cost EMI's etc.",
    2: "High Income, Low Spending Score (Cluster 3) - Target these customers by asking for feedback and advertising the product in a better way to convert them into Cluster 5 customers.",
    3: "Low Income, Low Spending Score (Cluster 4) - Don't target these customers since they have less income and need to save money.",
    4: "High Income, High Spending Score (Cluster 5) - Target these customers by sending new product alerts which would lead to an increase in the revenue collected by the mall as they are loyal customers."
}

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            annual_income = float(request.form['annual_income'])
            spending_score = float(request.form['spending_score'])

            # Standardize the user input using the same scaler
            user_input_scaled = scaler.transform([[annual_income, spending_score]])

            # Predict the cluster for the user input
            predicted_cluster = kmeans_model.predict(user_input_scaled)[0]

            # Get cluster information
            cluster_description = cluster_info.get(predicted_cluster, 'Unknown Cluster')

            return render_template('index.html', prediction=cluster_description)

        except Exception as e:
            error_message = f"Error: {str(e)}"
            return render_template('index.html', error=error_message)

if __name__ == '__main__':
    app.run(debug=True)
