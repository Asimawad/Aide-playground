import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Step 1: Read the data
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

# Step 2: Create a simple RandomForestClassifier model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Step 3: Train the model on the train_df
model.fit(train_df[["feature1", "feature2"]], train_df["target"])

# Step 4: Predict probabilities on the test_df
predictions = model.predict_proba(test_df[["feature1", "feature2"]])[:, 1]

# Step 5: Save the predicted probabilities into the 'submission.csv' file
submission_df = pd.DataFrame({"id": test_df["id"], "target": predictions})
submission_df.to_csv("./submission/submission.csv", index=False)


output
Done


import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Step 1: Read the data
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

# Step 2: Create a simple RandomForestClassifier model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Step 3: Train the model on the train_df
model.fit(train_df[["feature1", "feature2"]], train_df["target"])

# Step 4: Predict probabilities on the test_df
predictions = model.predict_proba(test_df[["feature1", "feature2"]])[:, 1]

# Step 5: Save the predicted probabilities into the 'submission.csv' file
submission_df = pd.DataFrame({"id": test_df["id"], "target": predictions})
submission_df.to_csv("./submission/submission.csv", index=False)


output
Done
