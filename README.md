# MLDA-project_Hotel-Reservations

This repository contains a machine learning and data analysis project focused on hotel reservation data. The main work is implemented in the notebook `hotel_reservations.ipynb`.

## Project objective

The project analyzes hotel booking records to understand which features influence whether a reservation is canceled or not. The notebook explores the dataset, visualizes patterns, and prepares the data for predictive modeling tasks.

The target variable is:

- `booking_status`: `Canceled` or `Not_Canceled`

## Dataset

The project uses a hotel reservations dataset that includes information such as:

- number of adults and children
- weekend and weekday nights
- meal plan type
- room type reserved
- lead time
- arrival month and date
- market segment type
- repeated guest indicator
- previous cancellation history
- average room price
- special requests
- reservation status

## Files

- `hotel_reservations.ipynb` — main notebook with exploratory analysis and model work
- `README.md` — repository documentation

## Setup

Create a Python environment and install the required dependencies:

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate   # Windows

pip install pandas seaborn matplotlib numpy scikit-learn jupyter
```

## Run the notebook

Start Jupyter Notebook:

```bash
jupyter notebook
```

Then open `hotel_reservations.ipynb` and run the cells in sequence.

## Workflow

The notebook typically includes the following steps:

1. Importing libraries
2. Loading the dataset
3. Checking dataset structure and missing values
4. Performing exploratory data analysis
5. Visualizing categorical and numerical distributions
6. Reviewing class balance in the target variable
7. Preparing features for modeling
8. Training and evaluating machine learning models

## Notes

The dataset is imbalanced, with a larger number of non-canceled bookings than canceled bookings. Because of this, it is important to consider evaluation metrics beyond accuracy, such as precision, recall, F1-score, and ROC-AUC when assessing model performance.

## Expected outcome

The goal of the project is to:

- analyze hotel reservation patterns
- identify factors related to cancellations
- build a machine learning workflow for prediction
- produce a final model for booking cancellation forecasting

## License

No explicit license file is currently included in the repository. If you plan to share or distribute the project publicly, consider adding a license such as MIT or Apache 2.0.

## Acknowledgements

This project is based on a hotel reservation dataset and was developed as part of the "Machine Learning and Data Analysis" coursework.
