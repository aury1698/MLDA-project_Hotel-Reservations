# MLDA-project_Hotel-Reservations

This repository contains a Machine Learning project aimed at predicting hotel booking cancellations using exploratory data analysis (EDA), feature selection, SMOTE class balancing, and hyperparameter optimization via GridSearchCV. The main work is implemented in the notebook `hotel_reservations.ipynb`.

Developed for the **Machine Learning** course at the **University of Perugia (UniPG)** – Department of Engineering.

---

## Project Overview

Booking cancellations represent a major challenge for the hospitality industry, affecting inventory allocation, pricing strategies, and revenue forecasting. This project leverages a dataset of **36,275 hotel reservations** with 19 initial attributes to build predictive models capable of determining whether a reservation will be canceled (`Canceled = 1`) or honored (`Not_Canceled = 0`).

By combining comprehensive **Exploratory Data Analysis (EDA)**, **One-Hot Encoding**, **SMOTE oversampling**, and **GridSearchCV hyperparameter tuning**, the workflow compares multiple classification architectures to maximize predictive performance (specifically optimizing for **F1-score** and **ROC-AUC**).

---

## Dataset & Key Insights

The project utilizes the `Hotel_Reservations.csv` dataset, comprising 36,275 reservation records.

### Key EDA Findings
1. **Target Class Imbalance**:
   - **Not Canceled (0)**: ~65% (~23,590 reservations)
   - **Canceled (1)**: ~35% (~12,685 reservations)
2. **Missing Values**: Clean dataset with **0 missing values** across all columns.
3. **Primary Predictors**:
   - **Lead Time (`lead_time`)**: The single strongest predictor of cancellation. Reservations made far in advance have a significantly higher probability of cancellation.
   - **Special Requests (`no_of_special_requests`)**: Guests who make special requests are much less likely to cancel.
   - **Parking Space (`required_car_parking_space`) & Repeated Guests (`repeated_guest`)**: Extremely low cancellation rates among returning guests or those requiring parking.
   - **Average Room Price (`avg_price_per_room`)**: Slightly higher cancellation rates observed for higher price tiers.
   - **Market Segment (`market_segment_type`)**: Online bookings represent the majority of reservations and exhibit distinct cancellation behaviors compared to corporate or offline bookings.

---

## Data Preprocessing & Pipeline

1. **Data Cleaning**:
   - Dropped non-informative identifier columns (`Booking_ID`).
   - Converted target labels: `Canceled` $\rightarrow 1$, `Not_Canceled` $\rightarrow 0$.
2. **Categorical Feature Encoding**:
   - Applied **One-Hot Encoding** (`pd.get_dummies` with `drop_first=True`) to convert categorical attributes (`type_of_meal_plan`, `room_type_reserved`, `market_segment_type`) into numerical format while preventing multi-collinearity.
3. **Feature Selection & Correlation Filtering**:
   - Evaluated feature correlations with `booking_status`.
   - Removed low-correlation / noisy features (e.g., `arrival_date`, `arrival_month`, low-frequency room and meal plan categories).
4. **Class Imbalance Resolution (SMOTE)**:
   - Utilized **SMOTE** (*Synthetic Minority Over-sampling Technique*) from `imblearn` to balance the dataset 50/50 prior to model training (expanding the training set to ~39,024 balanced samples), preventing model bias toward the majority class.
5. **Feature Standardization**:
   - Split data into **80% Training** and **20% Test** sets (`train_test_split`, `random_state=42`).
   - Applied `StandardScaler` to normalize feature ranges across all model inputs.

---

## Model Building & Hyperparameter Optimization

Model training and hyperparameter search were executed using a custom `Pipeline` wrapped inside `GridSearchCV` with **5-fold Cross-Validation (`cv=5`)**, specifically optimizing for the **F1-score**.

### Architectures & Search Space
* **Logistic Regression**:
  * Penalties: `L2`, `None`
  * Regularization strength $C$: Logarithmically spaced grid ($10^0$ to $10^5$)
* **Multi-Layer Perceptron (`MLPClassifier`)**:
  * Hidden Layer Architectures: `(50,)`, `(100,)`, `(50, 50)`, `(100, 100)`
  * Activation Functions: `tanh`, `relu`
  * Solvers: `adam`, `sgd`
  * L2 Regularization $\alpha$: $[10^{-4}, 10^{-3}, 0.01, 0.1, 1, 10, 100]$
* **K-Nearest Neighbors (`KNeighborsClassifier`)**:
  * $k$ Neighbors: $[3, 5, 7, 9]$
  * Weights: `uniform`, `distance`
  * Algorithm: `kd_tree`, `auto`
* **Support Vector Machines (`SVC`)**:
  * Kernels: `linear`, `rbf`, `poly` (Note: SVC was evaluated in specific search runs but omitted from long iterations due to high computational cost on large SMOTE sample sizes).

---

## Experimental Results & Performance Comparison

Multiple grid search experiments were conducted to evaluate model efficacy on the held-out test set ($N = 9,756$ test samples).

### 1. Neural Network Benchmark (Logistic Regression vs. MLP)
When restricting search to Logistic Regression and Multi-Layer Perceptron:
* **Best Model**: `MLPClassifier`
* **Optimal Hyperparameters**: `activation='tanh'`, `alpha=0.001`, `hidden_layer_sizes=(100, 100)`, `solver='adam'`
* **Training Accuracy / F1-Score**: **0.96**
* **Test Performance**:
  * **Test Accuracy**: **88.0%**
  * **Class 0 (Not Canceled)**: Precision: 0.87 | Recall: 0.88 | F1-Score: 0.88
  * **Class 1 (Canceled)**: Precision: 0.88 | Recall: 0.87 | F1-Score: 0.88
  * **Macro Avg F1-Score**: **0.88**

### 2. Multi-Model Benchmark (including KNN)
When expanding the search space to include K-Nearest Neighbors:
* **Best Overall Model**: `KNeighborsClassifier`
* **Optimal Hyperparameters**: `algorithm='kd_tree'`, `n_neighbors=9`, `weights='distance'`
* **Training Accuracy / F1-Score**: **0.99** (distance weighting)
* **Test Performance**:
  * **Test Accuracy**: **90.0%**
  * **Class 0 (Not Canceled)**: Precision: 0.90 | Recall: 0.89 | F1-Score: 0.89
  * **Class 1 (Canceled)**: Precision: 0.89 | Recall: 0.90 | F1-Score: 0.90
  * **Macro Avg F1-Score**: **0.90**

### Performance Summary Table

| Model Architecture | Best Parameters | Test Accuracy | Test Macro F1-Score |
| :--- | :--- | :---: | :---: |
| **MLPClassifier** | `tanh`, `hidden=(100,100)`, `alpha=0.001`, `adam` | **88.0%** | **0.88** |
| **KNeighborsClassifier** | `kd_tree`, `k=9`, `weights='distance'` | **90.0%** | **0.90** |

---

## Tech Stack & Requirements

* **Language**: Python 3.x
* **Core Libraries**:
  * Data Processing: `pandas`, `numpy`
  * Visualization: `seaborn`, `matplotlib`
  * Machine Learning: `scikit-learn` (`StandardScaler`, `LogisticRegression`, `MLPClassifier`, `KNeighborsClassifier`, `GridSearchCV`)
  * Imbalanced Learning: `imbalanced-learn` (`imblearn.over_sampling.SMOTE`)

### How to Run

1. Open `hotel_reservations.ipynb` in **Jupyter Notebook** or **Google Colab**.
2. Execute all cells sequentially to reproduce the EDA plots, SMOTE resampling, pipeline optimization, and final classification metrics.

---

## Authors

* **Aurora Traversini**

**Department of Engineering – University of Perugia (UniPG)**
