# Employee Attrition Prediction and Retention Strategy Analysis

## Project Objective

The primary goal of this project is to predict employee attrition (whether an employee is likely to leave the company) using historical HR data from the IBM HR Analytics Dataset. Beyond prediction, the project aims to understand the key factors driving attrition and translate these findings into actionable retention strategies for the Human Resources department.

## Dataset

*   **Source:** IBM HR Analytics Employee Attrition & Performance dataset.
*   **Content:** The dataset contains fictional HR records for 1470 employees, covering a wide range of attributes such as demographics, job role, satisfaction levels, compensation, tenure, performance ratings, and work-life balance factors.
*   **Target Variable:** `Attrition` (Categorical: 'Yes' or 'No'), indicating whether the employee left the company.

## Methodology

The project followed these key steps:

1.  **Data Loading & Initial Inspection:** The dataset (`WA_Fn-UseC_-HR-Employee-Attrition.csv`) was loaded into a pandas DataFrame. Basic checks were performed for shape, data types, missing values, and initial statistical summaries. Identified and removed irrelevant columns (`EmployeeNumber`, `EmployeeCount`, `StandardHours`, `Over18`).

2.  **Exploratory Data Analysis (EDA):**
    *   Analyzed the distribution of the target variable (`Attrition`), revealing a class imbalance (more 'No' than 'Yes').
    *   Visualized distributions of numerical features (histograms) and categorical features (count plots).
    *   Performed bivariate analysis comparing each feature against the `Attrition` variable using box plots (for numerical) and count plots (for categorical) to identify potential relationships.
    *   Calculated and visualized a correlation matrix for numerical features to understand linear relationships.
    *   Key hypotheses from EDA pointed towards factors like Overtime, Monthly Income, Job Level, Tenure (various metrics), Age, and Job Satisfaction as potential drivers.

3.  **Data Preprocessing:**
    *   Encoded the target variable `Attrition` into numerical format ('Yes': 1, 'No': 0).
    *   Separated features (X) and the target variable (y).
    *   Identified numerical and categorical features for distinct processing.
    *   Applied `StandardScaler` to numerical features to normalize their scale.
    *   Applied `OneHotEncoder` to categorical features to convert them into a numerical format suitable for the model, dropping the first category to avoid multicollinearity.
    *   Combined these steps using `ColumnTransformer` and integrated them into a `Pipeline`.
    *   Split the data into training (75%) and testing (25%) sets, ensuring stratification to maintain the original attrition ratio in both sets.

4.  **Model Training & Evaluation:**
    *   A **Logistic Regression** model was chosen for classification.
    *   The model was trained using a scikit-learn `Pipeline` that included the preprocessing steps.
    *   `class_weight='balanced'` was used to address the data imbalance, giving more importance to correctly classifying the minority class (Attrition='Yes').
    *   The trained model was evaluated on the unseen test set using:
        *   Accuracy
        *   ROC AUC Score
        *   Classification Report (Precision, Recall, F1-Score)
        *   Confusion Matrix
    *   The model demonstrated reasonable predictive performance, particularly in identifying potential attrition cases.

5.  **Model Explanation (SHAP):**
    *   The **SHAP (SHapley Additive exPlanations)** library, specifically the `LinearExplainer`, was used to interpret the predictions of the Logistic Regression model.
    *   SHAP values were calculated for the test set to quantify the contribution of each feature to individual predictions (on the log-odds scale).
    *   Visualizations were generated:
        *   **Summary Bar Plot:** Showed the average impact (importance) of each feature across all predictions.
        *   **Summary Dot/Beeswarm Plot:** Illustrated how the value of a feature (high or low) influences the direction and magnitude of its contribution to the attrition prediction.
        *   **Dependence Plots:** Examined the relationship between individual feature values and their SHAP values.
        *   **Force Plot (Example):** Demonstrated how different features pushed a single prediction towards or away from the baseline (average) prediction.

6.  **Deriving Actionable Insights:** The findings from EDA and, crucially, the SHAP explanations were synthesized into practical recommendations for HR.

## Key Findings & Model Insights (Based on Logistic Regression & SHAP)

SHAP analysis confirmed and quantified the importance of several factors driving the model's attrition predictions:

*   **Overtime:** Consistently emerged as one of the **most significant factors** increasing the likelihood of attrition.
*   **Compensation & Job Level:** Lower `MonthlyIncome` and lower `JobLevel` strongly increased predicted attrition risk.
*   **Tenure & Experience:** Shorter tenure metrics like `TotalWorkingYears`, `YearsAtCompany`, `YearsInCurrentRole`, and `YearsWithCurrManager` were associated with higher attrition probability.
*   **Age:** Younger employees (`Age`) generally showed a higher tendency to leave according to the model.
*   **Job & Environment Factors:** Lower `JobSatisfaction`, lower `EnvironmentSatisfaction`, and potentially lower `JobInvolvement` increased predicted risk.
*   **Marital Status:** Being `Single` was identified as a factor increasing attrition likelihood compared to Married or Divorced statuses.
*   **Other Potential Factors:** Depending on the specific model run, `DistanceFromHome`, `NumCompaniesWorked`, and specific `JobRole`s (after encoding) might also show noticeable influence.

## Actionable Insights for HR Retention Strategies

Based on the analysis, the following strategies are recommended to reduce employee attrition:

1.  **Manage Overtime Effectively:** Investigate reasons for high overtime, monitor workloads, review staffing levels, and promote a culture of work-life balance.
2.  **Enhance Compensation & Growth Opportunities:** Ensure competitive salaries (especially for lower levels/income roles), define clear career paths, provide skill development opportunities, and implement fair performance recognition.
3.  **Improve Manager Support & Engagement:** Train managers on people skills, implement robust onboarding, encourage regular 1-on-1s, and consider mentorship programs, particularly for newer employees.
4.  **Boost Job Satisfaction & Work Environment:** Actively solicit and act upon employee feedback regarding their job and workplace environment. Increase job autonomy and involvement where feasible.
5.  **Address Specific Group Needs:** Understand the unique challenges of high-attrition roles or demographic groups (e.g., frequent travelers, single employees, those with long commutes) and tailor support or policies (like hybrid work) accordingly.

## Technical Requirements

*   Python (e.g., 3.8+)
*   pandas
*   numpy
*   scikit-learn
*   matplotlib
*   seaborn
*   shap

## How to Run

The analysis was conducted using Python scripts/notebooks. Key steps involved:
1.  Loading the `WA_Fn-UseC_-HR-Employee-Attrition.csv` dataset.
2.  Executing the data cleaning and preprocessing steps.
3.  Training the Logistic Regression model using the defined pipeline.
4.  Evaluating the model performance using standard classification metrics.
5.  Applying the SHAP `LinearExplainer` to the trained model and preprocessed data to generate explanations and plots.

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/aneesanwaar/Predict-Employee-Attrition.git
    cd Predict-Employee-Attrition
    ```
    
2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  Ensure the dataset file (`WA_Fn-UseC_-HR-Employee-Attrition.csv`) is in the root directory or adjust the path in the code.
2.  Open and run the Jupyter Notebook:
    ```bash
    jupyter notebook employee_attrition_analysis.ipynb
    ```
    *(Or, if using a script: `python employee_attrition_script.py`)*
3.  Follow the steps within the notebook/script to perform the analysis, train the model, and generate explanations.


## Future Work

*   Experiment with other classification models (e.g., Random Forest, Gradient Boosting) and compare performance/interpretability.
*   Perform hyperparameter tuning on the chosen model(s) to potentially improve predictive accuracy.
*   Explore more advanced feature engineering techniques.
*   Develop a deployment strategy for the model to allow for real-time or batch prediction on new employee data.
*   Integrate qualitative data (e.g., detailed exit interview notes) for a richer understanding.
