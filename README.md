# Banking Analytics Suite: Customer Churn and Salary Prediction

A comprehensive deep learning solution implementing both binary classification for customer churn prediction and regression for salary estimation using TensorFlow and Neural Networks.

## Project Architecture
```
├── app.py                     # Churn prediction Streamlit interface
├── regression.py             # Salary prediction Streamlit interface
├── Churn_Modelling.csv      # Source dataset
├── experiments.ipynb        # Primary model development notebook
├── hyperparameter_tuning.ipynb # Model optimization notebook
├── regression.ipynb         # Salary prediction model notebook
├── prediction.ipynb         # Inference notebook
├── model.h5                 # Churn prediction model
├── regression_model.h5      # Salary prediction model
├── requirements.txt         # Project dependencies
├── preprocessing/
│   ├── scaler.pkl          # StandardScaler for feature normalization
│   ├── label_encoder_gender.pkl # Gender categorical encoder
│   └── onehot_encoder_geo.pkl  # Geography one-hot encoder
└── logs/
    ├── fit/                # Churn model TensorBoard logs
    └── regressionlogs/    # Salary model TensorBoard logs
```

## Technical Implementation

### Deep Learning Architecture

#### Churn Prediction Model
- **Architecture**: Sequential Neural Network
  - Input Layer: Dense(64, ReLU)
  - Hidden Layer: Dense(32, ReLU)
  - Output Layer: Dense(1, Sigmoid)
- **Optimizer**: Adam (lr=0.01)
- **Loss Function**: Binary Cross-Entropy
- **Metrics**: Accuracy
- **Early Stopping**: patience=20, monitor='val_loss'

#### Salary Prediction Model
- **Architecture**: Sequential Neural Network
  - Input Layer: Dense(64, ReLU)
  - Hidden Layer: Dense(32, ReLU)
  - Output Layer: Dense(1, Linear)
- **Loss Function**: Mean Absolute Error
- **Metrics**: MAE
- **Early Stopping**: patience=10

### Hyperparameter Optimization
- **Framework**: scikit-learn GridSearchCV
- **Parameters Tuned**:
  - Neurons: [16, 32, 64, 128]
  - Hidden Layers: [1, 2]
  - Epochs: [50, 100]
- **Cross-validation**: 3-fold
- **Parallel Processing**: Enabled (n_jobs=-1)

### Data Preprocessing Pipeline
- **Categorical Encoding**:
  - Geography: OneHotEncoder(handle_unknown='ignore')
  - Gender: LabelEncoder
- **Numerical Features**: StandardScaler
- **Train-Test Split**: 80-20 ratio
- **Random State**: 42

### Feature Set
- **Numerical Features**:
  - Credit Score
  - Age
  - Tenure
  - Balance
  - Number of Products
  - Estimated Salary
- **Categorical Features**:
  - Geography (France, Spain, Germany)
  - Gender
  - Has Credit Card
  - Is Active Member

## Installation

1. Create and activate virtual environment:
```bash
python -m venv env
source env/bin/activate  # Linux/Mac
env\Scripts\activate     # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Web Applications
Launch Churn Prediction interface:
```bash
streamlit run app.py
```

Launch Salary Prediction interface:
```bash
streamlit run regression.py
```

### Model Training
```python
# Churn Prediction
python experiments.ipynb

# Salary Prediction
python regression.ipynb

# Hyperparameter Optimization
python hyperparameter_tuning.ipynb
```

### TensorBoard Visualization
```bash
tensorboard --logdir logs/fit  # Churn model metrics
tensorboard --logdir regressionlogs/fit  # Salary model metrics
```

## Dependencies
```
tensorflow==2.15.0
pandas
numpy
scikit-learn
tensorboard
matplotlib
streamlit
keras
```

## Model Performance Monitoring
- **TensorBoard Integration**: Real-time monitoring of:
  - Loss curves
  - Accuracy metrics
  - Model weights/biases distributions
  - Learning rate progression
- **Validation Metrics**: 
  - Churn Prediction: Binary Accuracy, AUC-ROC
  - Salary Prediction: MAE, MSE

## Production Deployment
Both models are deployed via Streamlit, featuring:
- Interactive web interfaces
- Real-time predictions
- Automated preprocessing pipeline
- Input validation
- Prediction probability visualization (Churn model)
- Salary estimation with confidence intervals (Regression model)
