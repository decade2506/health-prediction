# Heart Failure Prediction Model

A machine learning project implementing logistic regression from scratch to predict mortality risk in heart failure patients using clinical data.

## About This Project

This project trains a custom logistic regression classifier on the [Heart Failure Clinical Records Dataset](https://www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data) to predict `DEATH_EVENT` from 12 patient features (age, anaemia, ejection fraction, serum creatinine, etc.).

**Key Features:**
- Custom `MyLogisticRegression` implementation using gradient descent
- Custom `MyScaler` for feature standardization
- Train/test split with configurable ratio
- Model evaluation with accuracy and classification report
- Prediction interface for single patient cases

## Setup & Installation

### Prerequisites
- Python 3.7+

### Install Dependencies

```bash
python -m pip install --upgrade pip
python -m pip install joblib pandas scikit-learn numpy matplotlib
```

## Project Structure

```
health-prediction/
├── README.md
├── heart_failure_clinical_records_dataset.csv  # Training dataset
├── logistic_model.pkl                         # (gitignored)Trained model
├── scaler.pkl                                 # (gitignored)Fitted scaler
└── src/
    ├── model.py                    # Train the model
    ├── predict.py                  # Make predictions on new data
    └── modules/
        ├── logistic.py             # MyLogisticRegression class
        ├── scaler.py               # MyScaler class
        └── testSplit.py            # Train/test split utility
```

## Usage

### 1. Train the Model

```bash
cd src
python model.py
```

This will:
- Load the CSV dataset
- Split into train (70%) / test (30%)
- Standardize features
- Train the logistic regression model
- Print accuracy and classification report
- Save `logistic_model_2.pkl` and `scaler_2.pkl`

### 2. Make Predictions

```bash
cd src
python predict.py
```

Follow the interactive prompts to enter patient clinical values. The model will return:
- Predicted class (Sống/Tử vong)
- Probability of death event

## API

### `MyLogisticRegression`
- `logisticRegression(X, Y)` - Train the model
- `predict(X)` - Return class predictions (0 or 1)
- `predict_proba(X)` - Return probabilities as array of shape (n_samples, 2)

### `MyScaler`
- `fit_transform(X)` - Standardize training data
- `transform(X)` - Standardize new data using learned mean/std

### `testSplit`
- `train_test(X, Y, test_size=0.3)` - Split lists/arrays `X` and `Y` into training and test sets. Returns `(X_train, Y_train, X_test, Y_test)` where `test_size` is the fraction reserved for testing. This mirrors `sklearn.model_selection.train_test_split` behavior used in the project.
## Repository Maintenance

- ✅ **Models (`*.pkl`) are NOT tracked.** They are in `.gitignore` to keep the repo clean.
- ✅ Always commit `src/model.py` — this is the source of truth for reproducibility.
- ✅ To regenerate models: run `python src/model.py` locally.
- ✅ To share trained models, use release assets, Git LFS, or artifact storage (S3, Hugging Face, etc).

## References

- **Scaler implementation:** [StandardScaler explanation](https://stackoverflow.com/questions/40758562/can-anyone-explain-me-standardscaler)
- **Logistic Regression:** Various ML texts and tutorials
- **Train/Test Split:** [sklearn.model_selection.train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
- **Dataset:** [Kaggle Heart Failure Clinical Records](https://www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data)
