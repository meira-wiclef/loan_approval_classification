from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

# loading and preparing the dataset
csv_filename = "loan_data.csv"
raw_dataset = pd.read_csv(csv_filename)  # this is a pandas dataframe

income_column = "person_income"  # feature 1
loan_amount_column = "loan_amnt"  # feature 2
target_column = "loan_status"  # the column we are trying to predict
feature_columns = [income_column, loan_amount_column]  # this is a list of strings

cleaned_dataset = raw_dataset.copy()
for col in feature_columns + [target_column]:  # list concatenation, so we can clean all 3 columns at once
    # converts each column to numbers and converts any invalid ones to NaN
    cleaned_dataset[col] = pd.to_numeric(cleaned_dataset[col], errors="coerce")

# dropping invalid rows and keeping a small sample of data
loan_dataset = cleaned_dataset.dropna(subset=feature_columns + [target_column]).head(250)
total_records = len(loan_dataset)

print("Usable loan records:", total_records)

# splitting into training data and test data
training_count = int(0.8 * total_records)
training_records = loan_dataset.iloc[:training_count]
testing_records = loan_dataset.iloc[training_count:]

training_percentage = (len(training_records) / total_records) * 100
testing_percentage = (len(testing_records) / total_records) * 100

print("Training set:", len(training_records), "records (" + str(round(training_percentage)) + "%)",)
print("Testing set:", len(testing_records), "records (" + str(round(testing_percentage)) + "%)",)

train_features = training_records[feature_columns].values.tolist()  # 2D array of applicants
train_targets = training_records[target_column].astype(int).tolist()  # 1D array of loan statuses

test_features = testing_records[feature_columns].values.tolist()
test_targets = testing_records[target_column].astype(int).tolist()


def calculate_euclidean_distance(first_point, second_point):
    first_array = np.array(first_point)
    second_array = np.array(second_point)
    squared_differences = (first_array - second_array) ** 2
    return np.sqrt(np.sum(squared_differences))


def get_distance_from_record(distance_record):
    return distance_record[0]


def predict_knn_label(train_features, train_targets, target_point, num_neighbors):
    neighbor_distances = []

    for index in range(len(train_features)):
        distance = calculate_euclidean_distance(target_point, train_features[index])
        actual_label = train_targets[index]
        neighbor_distances.append((distance, actual_label))

    neighbor_distances.sort(key=get_distance_from_record)

    nearest_records = neighbor_distances[:num_neighbors]
    nearest_labels = []
    for distance, label in nearest_records:
        nearest_labels.append(label)

    label_frequency = Counter(nearest_labels)
    most_common_label = label_frequency.most_common(1)[0][0]
    return most_common_label


# evaluating accuracy with scikit learn
k = 5
predicted_targets = []

for current_test_point in test_features:
    predicted_label = predict_knn_label(train_features, train_targets, current_test_point, k)
    predicted_targets.append(predicted_label)

model_accuracy = accuracy_score(test_targets, predicted_targets)
print("Model Accuracy:", round(model_accuracy, 3))

# plotting the testing and training data
status_names = {0: "Rejected (0)",
                1: "Approved (1)"
                }

plt.figure(figsize=(8, 5))

unique_labels = sorted(training_records[target_column].unique())  # sorts [0,1] in ascending
for label in unique_labels:
    int_label = int(label)
    matching_rows = training_records[training_records[target_column] == label]
    readable_label = status_names.get(int_label, "Status " + str(int_label))
    plt.scatter(
        matching_rows[income_column],
        matching_rows[loan_amount_column],
        label="Train: " + readable_label,
        alpha=0.75,
    )

plt.scatter(
    testing_records[income_column],
    testing_records[loan_amount_column],
    color="black",
    marker="X",
    s=90,
    label="Test points",
)

plt.title("KNN Loan Approval Classification (k = " + str(k) + ")")
plt.xlabel("Annual Income (person_income)")
plt.ylabel("Loan Amount (loan_amnt)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
