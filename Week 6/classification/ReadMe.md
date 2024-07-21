# Weather Classification Model

This project aims to build a weather classification model using various machine learning algorithms, including Logistic Regression, Decision Tree, and Random Forest. The dataset contains various weather-related features, and the goal is to classify the weather type accurately.

## Project Structure

1. **Import Necessary Libraries**: The essential libraries for data manipulation, visualization, and machine learning are imported.

2. **Load the Dataset**: The weather classification dataset is loaded using `pandas`.

3. **Handle Missing Values**: Missing values in the dataset are handled using median imputation.

4. **Convert Categorical Variables**: Categorical variables are converted to numerical variables using one-hot encoding.

5. **Normalize/Standardize Numerical Features**: Numerical features are normalized to ensure that the models perform optimally.

6. **Split the Data**: The dataset is split into training and testing sets to evaluate the performance of the models.

7. **Perform Exploratory Data Analysis (EDA)**: Various plots and statistical summaries are generated to understand the dataset better.

8. **Apply Machine Learning Models**: 
    - **Logistic Regression**: A logistic regression model is trained with hyperparameter tuning.
    - **Decision Tree**: A decision tree classifier is trained with hyperparameter tuning.
    - **Random Forest**: A random forest classifier is trained with hyperparameter tuning.

9. **Evaluate Model Performance**: The performance of each model is evaluated based on accuracy, precision, recall, and f1-score.

10. **Compare Model Performance**: The models are compared to determine which one performs the best for weather classification.

## Key Findings

- The Random Forest classifier achieved the highest accuracy of 91.48%, followed by the Decision Tree classifier with 91.21%, and Logistic Regression with 85.11%.
- Both Random Forest and Decision Tree models outperformed Logistic Regression across all performance metrics.
- For practical implementation, the Random Forest classifier is recommended due to its robustness and higher accuracy in classifying weather types.

## Conclusion

The Decision Tree and Random Forest classifiers are more effective for weather classification in this dataset. The Random Forest classifier is slightly better and is recommended for practical applications due to its higher accuracy and overall performance.



