# ```python
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

# Load data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Preprocess data: handle missing values and encode categorical variables
def preprocess(data):
    # Impute missing values with mean for numerical columns
    num_cols = data.select_dtypes(include='number').columns
    data[num_cols] = data[num_cols].fillna(data[num_cols].mean())
    
    # Convert categorical columns to dummy variables
    cat_cols = data.select_dtypes(exclude='number').columns
    data = pd.get_dummies(data, drop_first=True, columns=cat_cols)
    return data

train_processed = preprocess(train)
test_processed = preprocess(test)

# Separate target variable and features for training
y_train = train['SalePrice']
X_train = train_processed.drop(columns=['Id', 'SalePrice'])
X_test = test_processed.drop(columns=['Id'])

# Split into K-Fold cross-validation sets
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Train XGBoost model with appropriate hyperparameters
params = {
    'n_estimators': 100,
    'max_depth': 3,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42
}
model = XGBRegressor(**params)
model.fit(X_train, y_train)

# Generate predictions on the test set
preds = model.predict(X_test)

# Save predictions to submission.csv
submission = pd.DataFrame({
    'Id': test['Id'],
    'SalePrice': preds
})
submission.to_csv('submission.csv', index=False)

# Calculate and print RMSE
from sklearn.metrics import mean_squared_error
import numpy as np

rmse = np.sqrt(mean_squared_error(np.log(y_train), model.predict(X_train)))
print(f"RMSE on training set: {rmse:.4f}")
