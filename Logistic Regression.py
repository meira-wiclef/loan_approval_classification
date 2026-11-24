import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# loading the dataset
csv_filename = "loan_data.csv"
raw_dataset = pd.read_csv(csv_filename)

income_column = "person_income"
loan_amount_column = "loan_amnt"
target_column = "loan_status"
feature_columns = [income_column, loan_amount_column]

cleaned_dataset = raw_dataset.copy()
for col in feature_columns + [target_column]:
    cleaned_dataset[col] = pd.to_numeric(cleaned_dataset[col], errors="coerce")  # invalid set as NaN

loan_dataset = cleaned_dataset.dropna(subset=feature_columns + [target_column]).head(250)
total_records = len(loan_dataset)

print("Usable loan records:", total_records)


training_count = int(0.8 * total_records)
training_records = loan_dataset.iloc[:training_count]
testing_records = loan_dataset.iloc[training_count:]

training_percentage = (len(training_records) / total_records) * 100
testing_percentage = (len(testing_records) / total_records) * 100

print("Training set:", len(training_records), "records (" + str(round(training_percentage)) + "%)",)
print("Testing set:", len(testing_records), "records (" + str(round(testing_percentage)) + "%)", )

train_features = training_records[feature_columns].values.tolist()
train_targets = training_records[target_column].astype(int).tolist()

test_features = testing_records[feature_columns].values.tolist()
test_targets = testing_records[target_column].astype(int).tolist()

# training the logistic model
logistic_model = LogisticRegression()
logistic_model.fit(train_features, train_targets)  # it's learning the weights of both features and the intercept

predicted_targets = logistic_model.predict(test_features).tolist()

model_accuracy = accuracy_score(test_targets, predicted_targets)
print("Model Accuracy:", round(model_accuracy, 2))
print("Learned weights (w1, w2):", logistic_model.coef_[0])
print("Intercept (b):", logistic_model.intercept_[0])

# plotting
status_names = {0: "Rejected (0)", 1: "Approved (1)"}  # what we group each point by

plt.figure(figsize=(8, 5))

unique_labels = sorted(training_records[target_column].unique())  # returns [0,1] sorted in ascending order
for label in unique_labels:
    int_label = int(label)
    matching_rows = training_records[training_records[target_column] == label]  # finds the matching rows for each label
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

plt.title("Logistic Regression Loan Approval Classification")
plt.xlabel("Annual Income (person_income)")
plt.ylabel("Loan Amount (loan_amnt)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
