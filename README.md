# **Turbofan Engine Failure Prediction**  

## **Project Overview**  
This project aims to predict the Remaining Useful Life (RUL) of turbofan engines using **machine learning models**. 
The dataset used comes from the **NASA Turbofan Engine Degradation Simulation Dataset**, 
which contains sensor readings from multiple engines over time. 
The goal is to develop a predictive model that can estimate how long an engine will continue to function before failure occurs.  

## **Things Learned**
- MLP and XGBoost performed the best in the experiments. 
- Ensemble methods improved predictive accuracy by combining multiple models. 
- Feature selection played a crucial role in optimizing performance.

## **Dataset Description**  
The dataset consists of time-series sensor readings for multiple engines. 
Each row corresponds to an engine at a specific time cycle, with features including operational settings and sensor measurements.  

### **Features**  
- `unit number` → Unique ID for each engine.  
- `time cycles` → Time in operational cycles.  
- `settings 1, settings 2, settings 3` → Operational settings.  
- `sensor 1` to `sensor 21` → Sensor readings from different parts of the engine.  
- `RUL` (Remaining Useful Life) → The target variable (for training).  

## **Methods Used**  
### **1) Exploratory Data Analysis (EDA)**  
- Visualized sensor readings over time.  
- Checked correlations between sensors and RUL.  
- Identified missing values and outliers.  

### **2) Feature Engineering**  
- Standardized numerical features using **StandardScaler**.  
- Selected most important sensors based on feature importance.  

### **3) Machine Learning Models**  
The implemented multiple models which were then compared based on 
their performance:  
- **Linear Regression** → Baseline model.  
- **Polynomial Regression** → Added non-linearity to improve predictions.  
- **Support Vector Regression (SVR)** → Captured complex relationships.  
- **Random Forest Regression** → Handled non-linear interactions well.  
- **Multi-Layer Perceptron (MLP) Neural Network** → Deep learning approach.

### 4) **Ensemble Learning**  
To improve performance, ensemble learning techniques were also used:  
- **Bagging (Random Forest)**  
- **Boosting (XGBoost, Gradient Boosting)**  


## **Model Evaluation**  
Each model was evaluated using the following metrics:  
- **Mean Absolute Error (MAE)**  
- **Mean Squared Error (MSE)**  
- **R² Score**  

| Model                 | Train MAE | Test MAE | Train R² | Test R² |
|-----------------------|-----------|----------|----------|---------|
| Linear Regression     | 17.58     | 19.51    | 0.73     | 0.66    |
| Polynomial Regression | 14.36     | 17.18    | 0.79     | 0.71    |
| SVM Regression        | 11.54     | 17.07    | 0.80     | 0.70    |
| Random Forest         | 4.93      | 14.81    | 0.97     | 0.77    |
| XGBoost               | 6.07      | 15.74    | 0.95     | 0.74    |
| MLP (Neural Network)  | 11.05     | 17.11    | 0.79     | 0.69    |

