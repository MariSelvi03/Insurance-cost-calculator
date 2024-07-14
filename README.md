# Health Insurance Cost Predictor

This project implements regression models to predict health insurance costs based on individual attributes like age, sex, BMI, smoker status, and region. It includes data preprocessing, exploratory analysis, and the application of various regression techniques such as linear, ridge, and lasso regression. Evaluation metrics like Mean Absolute Error (MAE) are used to assess model performance, with results visualized through scatter plots comparing predicted versus actual expenses.

## Project Setup and Structure

### Installation Instructions

1. **Clone the repository**:
    ```sh
    git clone https://github.com/yourusername/health-insurance-cost-predictor.git
    cd health-insurance-cost-predictor
    ```

2. **Create and activate a virtual environment**:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install the dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

### Usage Instructions

1. **Download the dataset**:
    The script automatically downloads the dataset, but ensure you have internet access for the first run.

2. **Run the Python script**:
    ```sh
    python linear_regression_health_cost_calculator.py
    ```

### Explanation of Files and Folders

- **images/**: This folder contains the output images from the analysis and model predictions.
  - `boxplot.png`: Visualizes the distribution of expenses to check for outliers.
  - `linear_regression.png`: Shows the true values vs predictions for linear regression.
  - `ridge_regression.png`: Shows the true values vs predictions for ridge regression.
  - `lasso_regression.png`: Shows the true values vs predictions for lasso regression.

- **linear_regression_health_cost_calculator.py**: The main script for data processing, model training, prediction, and visualization.

- **requirements.txt**: Lists all the Python packages required to run the project.

- **README.md**: Provides an overview of the project, installation instructions, usage instructions, and explanations of the files and folders.

### Additional Information

- **insurance.csv**: The dataset used in this project is publicly available and downloaded from FreeCodeCamp.

- **Dependencies**: 
  - `pandas`: For data manipulation and analysis.
  - `scikit-learn`: For machine learning algorithms and data preprocessing.
  - `matplotlib` and `seaborn`: For data visualization.

---
