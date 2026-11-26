from collections import Counter  # counts how often each class label appears
import matplotlib.pyplot as plt  # used to plot a graph of the dataset
import numpy as np  # used for vector squaring
import pandas as pd  # used to extract csv
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, confusion_matrix  # evaluates accuracy
from sklearn.preprocessing import StandardScaler  # standardizes features so larger scales don't dominate distance

# 1. loading and preparing the dataset
csv_filename = "loan_data.csv"
raw_dataset = pd.read_csv(csv_filename)  # this is a pandas dataframe

income_column = "person_income"  # feature 1
loan_amount_column = "loan_amnt"  # feature 2
credit_score_column = "credit_score"  # feature 3
target_column = "loan_status"  # the column we are trying to predict
feature_columns = [income_column, loan_amount_column, credit_score_column]  # this is a list of strings

cleaned_dataset = raw_dataset.copy()
for col in feature_columns + [target_column]:  # list concatenation, so we can clean all 4 columns at once
    # converts each column to numbers and converts any invalid ones to NaN
    cleaned_dataset[col] = pd.to_numeric(cleaned_dataset[col], errors="coerce")

# dropping invalid rows and keeping a small sample of data
loan_dataset = cleaned_dataset.dropna(subset=feature_columns + [target_column]).head(600).sample(frac=1, random_state=42)
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

raw_train_features = training_records[feature_columns].values.tolist()  # 2D array of applicants
train_targets = training_records[target_column].astype(int).tolist()  # 1D array of loan statuses

raw_test_features = testing_records[feature_columns].values.tolist()
test_targets = testing_records[target_column].astype(int).tolist()

# fitting StandardScaler on training features only to avoid data leakage
scaler = StandardScaler()
train_features = scaler.fit_transform(raw_train_features).tolist()
test_features = scaler.transform(raw_test_features).tolist()


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


status_names = {0: "Rejected (0)",
                1: "Approved (1)"
}

# 2. using elbow method to test for best k
k_values = list(range(1, 9))
accuracies = []

for k in k_values:
    predicted_targets = [
        predict_knn_label(train_features, train_targets, test_point, k)
        for test_point in test_features
    ]
    accuracy = accuracy_score(test_targets, predicted_targets)
    accuracies.append(accuracy)

best_k = k_values[int(np.argmax(accuracies))]
print("Best k from elbow evaluation:", best_k)

plt.figure(figsize=(8, 5))
plt.plot(k_values, accuracies, marker="o")
plt.title("KNN Accuracy Across Different Values of k")
plt.xlabel("Number of Neighbors (k)")
plt.ylabel("Accuracy")
plt.xticks(k_values)
plt.grid(True)
plt.tight_layout()
plt.show()

# 3. evaluating accuracy with scikit learn and plotting right vs. wrong predictions
best_predicted_targets = [
    predict_knn_label(train_features, train_targets, current_test_point, best_k)
    for current_test_point in test_features
]
model_accuracy = accuracy_score(test_targets, best_predicted_targets)
print(f"Model Accuracy (k = {best_k}):", round(100 * model_accuracy, 3))

cm = confusion_matrix(test_targets, best_predicted_targets, labels=[0, 1])
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[status_names[0], status_names[1]],
)
fig, ax = plt.subplots(figsize=(7, 5))
disp.plot(ax=ax, cmap="Blues", values_format="d")
ax.set_title(f"KNN Confusion Matrix (k = {best_k})")
plt.tight_layout()
plt.show()

# 4. entering new data to determine its loan status
try:
    custom_income = float(input("\nEnter annual income (person_income): "))
    custom_loan_amnt = float(input("Enter loan amount (loan_amnt): "))
    custom_credit_score = float(input("Enter credit score (credit_score): "))
    custom_point = [custom_income, custom_loan_amnt, custom_credit_score]
    scaled_custom_point = scaler.transform([custom_point])[0].tolist()
    # predicting
    custom_prediction = predict_knn_label(train_features, train_targets, scaled_custom_point, best_k)
    readable_prediction = status_names.get(custom_prediction, f"Status {custom_prediction}")
    print(f"Predicted loan_status for income={custom_income}, loan_amnt={custom_loan_amnt}, credit_score={custom_credit_score}: {readable_prediction}")
except ValueError:
    print("Please enter valid numeric values for income, loan amount, and credit score.")