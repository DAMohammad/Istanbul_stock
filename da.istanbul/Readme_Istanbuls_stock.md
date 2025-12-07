# Istanbul Stock Exchange Data Regression This project is a simple **linear regression** exercise performed on Istanbul Stock Exchange data. The goal of this project is to predict the stock market value using various features. **Linear Regression** is used to predict the target value. ## Project Contents: - Loading data from a CSV file - Data preprocessing steps including: - Converting columns to numeric types - Removing **NaN** values - Standardizing data - Removing **outliers** using the **IQR** method - Handling non-logical values (if any) - Building a linear regression model using standardized data - Evaluating the model using **MSE (Mean Squared Error)** and **RMSE (Root Mean Squared Error)** ## Requirements: - Python 3.x - Required Libraries: - pandas - numpy - scikit-learn - matplotlib - seaborn You can install the required libraries using pip: `bash pip install pandas numpy scikit-learn matplotlib seaborn 
Usage:
First, place the data file at ../Database_Regressions/data_akbilgic.csv.
Then, run the code to build the linear regression model.
Finally, the MSE and RMSE of the model will be printed.
Code Explanation:
Loading Data:
Data is read from a CSV file, and preprocessing such as converting columns to numeric types is performed.
Removing Outliers:
Outliers are identified and removed using the IQR (Interquartile Range) method.
Standardizing Data:
Data is standardized using StandardScaler so that the features are on a similar scale.
Building the Linear Regression Model:
A linear regression model is trained on the standardized data and predictions are made on the test data.
Model Evaluation:
The model is evaluated using MSE and RMSE. These values indicate how accurate the model is.
Example Output:
MSE: 0.0007942890629459026 RMSE: 0.02818313437050433 
Challenges and Notes:
The data contains outliers, which may affect model performance. In this project, we attempt to remove them using the IQR method, though improvements may still be needed.
This project uses a simple linear regression model. More advanced models like Decision Trees or Neural Networks could be used for better predictions.
How to Run the Code:
To run the code, first download the file and then navigate to the folder where the data_akbilgic.csv file is located and run the following:
python data_akbilgic.py 
Future Enhancements:
Adding the ability to compare models and using more complex models like Decision Trees.
Implementing cross-validation for better model evaluation.
Investigating and refining the outlier removal process with different techniques.
Developers:
This project was created by [DAMohammad].
