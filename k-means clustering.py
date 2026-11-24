import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

csv_filename = "loan_data.csv"
raw_dataset = pd.read_csv(csv_filename)

income_column = "person_income"
loan_amount_column = "loan_amnt"
target_column = "loan_status"
feature_columns = [income_column, loan_amount_column]

cleaned_dataset = raw_dataset.copy()
for col in feature_columns + [target_column]:
    cleaned_dataset[col] = pd.to_numeric(cleaned_dataset[col], errors="coerce")

loan_dataset = cleaned_dataset.dropna(subset=feature_columns + [target_column]).head(250)
total_records = len(loan_dataset)

print("Usable loan records:", total_records)

feature_points = loan_dataset[feature_columns].values.tolist()  # converts to 2D array for applicants
actual_statuses = loan_dataset[target_column].astype(int).tolist()  # won't be used by the model, for comparison

num_clusters = 2  # 0 or 1 status
random_seed = 42  # so it generates the same sequence of clusters

kmeans_model = KMeans(n_clusters=num_clusters, random_state=random_seed, n_init=10)
cluster_labels = kmeans_model.fit_predict(feature_points).tolist()  # returns each rows cluster id
centroids = kmeans_model.cluster_centers_

# counting how many loan records belong to each cluster
cluster_counts = {}
for cluster_id in range(num_clusters):
    cluster_counts[cluster_id] = cluster_labels.count(cluster_id)
print("Cluster counts:", cluster_counts)

print("Cluster centroids (person_income, loan_amnt):")
for i, center in enumerate(centroids):
    print("  Cluster", i, "->", round(center[0], 2), round(center[1], 2))

plt.figure(figsize=(8, 5))

for cluster_id in range(num_clusters):
    cluster_incomes = []
    cluster_loans = []

    for index in range(total_records):
        if cluster_labels[index] == cluster_id:
            cluster_incomes.append(feature_points[index][0])
            cluster_loans.append(feature_points[index][1])

    plt.scatter(
        cluster_incomes,
        cluster_loans,
        label="Cluster " + str(cluster_id),
        alpha=0.75,
    )

plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    color="black",
    marker="X",
    s=140,
    label="Centroids",
)

plt.title("K-Means Loan Clustering (k = " + str(num_clusters) + ")")
plt.xlabel("Annual Income (person_income)")
plt.ylabel("Loan Amount (loan_amnt)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
