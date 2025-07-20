# Customer Churn Prediction

A deep learning-based solution for predicting customer churn using TensorFlow and Neural Networks. This project implements a binary classification model to predict whether a customer is likely to leave a bank's services based on various customer attributes.

## Project Structure
```
├── app.py                     # Streamlit web application
├── Churn_Modelling.csv       # Dataset file
├── experiments.ipynb         # Model development and training notebook
├── prediction.ipynb          # Model inference notebook
├── model.h5                  # Trained neural network model
├── requirements.txt          # Project dependencies
├── scaler.pkl               # StandardScaler for feature scaling
├── label_encoder_gender.pkl # Label encoder for gender feature
├── onehot_encoder_geo.pkl   # OneHot encoder for geography feature
└── logs/                    # TensorBoard logging directory
```

## Technical Stack
- **Deep Learning Framework**: TensorFlow 2.15.0
- **Model Architecture**: Sequential Neural Network with:
  - Input Layer: Dense(64, activation='relu')
  - Hidden Layer: Dense(32, activation='relu')
  - Output Layer: Dense(1, activation='sigmoid')
- **Optimizer**: Adam (learning_rate=0.01)
- **Loss Function**: Binary Cross-Entropy
- **Preprocessing**: StandardScaler, LabelEncoder, OneHotEncoder
- **Web Framework**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Model Monitoring**: TensorBoard

## Features
- Binary classification for customer churn prediction
- Real-time prediction through web interface
- Automated feature preprocessing
- Model performance monitoring via TensorBoard
- Support for categorical (Geography, Gender) and numerical features

## Model Features
- Credit Score
- Geography (France, Spain, Germany)
- Gender
- Age
- Tenure
- Balance
- Number of Products
- Has Credit Card
- Is Active Member
- Estimated Salary

## Installation
1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
### Web Application
Run the Streamlit app:
```bash
streamlit run app.py
```

### Jupyter Notebooks
- `experiments.ipynb`: Model development, training, and evaluation
- `prediction.ipynb`: Sample inference code

## Model Training
The model implements early stopping with patience=20 and monitors validation loss for optimal performance. Training metrics are logged using TensorBoard for visualization and monitoring.

## Performance Monitoring
View training metrics:
```bash
tensorboard --logdir logs/fit
```

## Data Preprocessing
- Categorical encoding:
  - Gender: Label Encoding
  - Geography: One-Hot Encoding
- Numerical features: Standard Scaling
- Feature engineering: Automated in preprocessing pipeline

## Dependencies
- tensorflow==2.15.0
- pandas
- numpy
- scikit-learn
- tensorboard
- matplotlib
- streamlit

## Production Deployment
The model is deployed using Streamlit, providing an interactive web interface for real-time predictions. All necessary preprocessing steps are integrated into the prediction pipeline.
