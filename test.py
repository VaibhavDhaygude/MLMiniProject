import pickle
import joblib

# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()

with open('model.pkl', 'rb') as model_file:
    kmeans_model = pickle.load(model_file)

scaler_filename = "scaler.save"
scaler = joblib.load(scaler_filename) 


# Assume you have a single new input point in the format [Annual Income, Spending Score]
new_input = [[200, 60]]  # Example single new input point

# Standardize the new input using the same scaler
new_input_scaled = scaler.transform(new_input)

# Predict the cluster for the new input
predicted_cluster = kmeans_model.predict(new_input_scaled)[0]

# Mapping of cluster numbers to descriptions and recommendations
cluster_info = {
    0: "Low Income, High Spending Score (Cluster 1) - May or may not target these group of customers based on the policy of the mall.",
    1: "Average Income, Average Spending Score (Cluster 2) - Can target these set of customers by providing them with Low cost EMI's etc.",
    2: "High Income, Low Spending Score (Cluster 3) - Target these customers by asking for feedback and advertising the product in a better way to convert them into Cluster 5 customers.",
    3: "Low Income, Low Spending Score (Cluster 4) - Don't target these customers since they have less income and need to save money.",
    4: "High Income, High Spending Score (Cluster 5) - Target these customers by sending new product alerts which would lead to an increase in the revenue collected by the mall as they are loyal customers."
}

# Print the information for the predicted cluster
print(cluster_info[predicted_cluster])
