# Evaluation Metrics for Regression Models
The most commonly-used metrics to evaluate the performance of regression models.

### MAE: Mean Absolute Error

Average value of the total absolute differences between the predicted values output by the model and the actual values in the dataset. It is expressed in the same unit scale as the data measured. Values closer to 0 are considered better.

Function to compute the MAE:
```
  def custom_mae(y_true, y_pred):
    absolute_sum = 0 # Initiating a variable for the accumulation of the absolute differences 
    
    # Iterating over each data point in both y_true and y_pred
    for true, predicted in zip(y_true, y_pred):
        
        # Subtracting predicted value from the true value to obtain the difference
        absolute_error = true - predicted
        
        # Obtaining the absolute value
        # If the difference is below 0, 
        if absolute_error < 0:
            absolute_error = -absolute_error # We make it positive by obtaining its negation{ (-)-n = +n }
        
        # We add the absolute error value to the current absolute sum value 
        absolute_sum += absolute_error
        
    # After iterating through every data point, we divide the absolute_sum by the total number of data values in y_true    
    mae = absolute_sum / len(y_true)
    
    return mae # Returning value
  ```
```
  # Using custom function
print('\nMean Absolute Error\n')
round(custom_mae(y_true, y_pred), 3)
```
```
Out[5]:

Mean Absolute Error
1.155
```
On average, the prediction is wrong by +- 1.155 units. So if a data point is equal to 100, is expected the predictions ouput by the model for this specific data point to fluctuate from 98.845 to 101.555.


### MSE: Mean Squared Error
```
def custom_rmse(y_true, y_pred):
    
    squared_sum = 0 # Initiating a squared sum variable equal to 0
    
    # Iterating over y_true and y_pred
    for true, predicted in zip(y_true, y_pred):
        
        # Subtracting predicted from true and squaring the result 
        squared_error = (true - predicted)**2
        
        # Adding the squared error result to the squared_sum variable
        squared_sum += squared_error
        
    # Obtaining the MSE by dividing the squared sum to the total number of data points in y_true    
    mse = squared_sum / len(y_true)
    
    # To find the square root, we raise the mse to the power of 0.5
    rmse = mse**0.5
    
    return rmse # Returning result
```
```
# Using custom function
print('\nRoot Mean Squared Error\n')
round(custom_rmse(y_true, y_pred), 3)
```
```
Root Mean Squared Error
1.282
```


The predictions are off by +- 1.282 units, especially when give more weights to larger errors.

### Maximum error
This is a good metric to know what is the worst-case scenario in validation set, the worst deviation between ```y_true```and ```y_pred```.
Lower values are preferred over large values.

```
def custom_max_error(y_true, y_pred):
    
    # Creating an empty list of absolute errors
    absolute_errors = []
    
    # Iterating through actual and predicted values for y
    for true, predicted in zip(y_true, y_pred):
        
        # Computing the differences(i.e., errors)
        error = true - predicted
        # Obtaining the absolute value
        if error < 0:  # If the difference is a negative number,
            error = -error # We obtain the negative of the negative, which is a positive number
        
        absolute_errors.append(error) # Adding absolute value to the list of empty errors
    
    # Obtaining the largest error in the absolute_errors list using the max() function
    maximum_error = max(absolute_errors)
    
    return maximum_error
```
```
# Using custom function
print('\nMaximum Error\n')
round(custom_max_error(y_true, y_pred), 3)
```
```
Maximum Error
2.2
```

In worst-case scenario, the largest deviation between the model's output and the actual values was +- 2.2 units. This is expressed in the same unit value as the observed data.


### Mean Absolute Percentage Error
It can be used for time-series data, such as forecasting sales or predicting the price of financial assets. It is expressed in percentage, making it easier to understand its results.
```
def custom_mape(y_true, y_pred):
    
    # Intiating an empty variable for the sum of absolute errors
    sum_absolute_errors = 0
    
    # Iterating over true and predicted values
    for actual, predicted in zip(y_true, y_pred):
        # Computing the differences between them
        absolute_error = actual - predicted
        
        # Obtaining the absolute value
        # If any number is below 0, we obtain the negative of this number to make it positive
        if absolute_error < 0:
            absolute_error = -absolute_error
        # We do the same for the value in y_true
        absolute_actual = actual
        if absolute_actual < 0:
            absolute_actual = -absolute_actual
            
        # We divide the absolute error by the absolute value of y_true
        absolute_error = absolute_error / absolute_actual  
        
        # We sum the values in absolute_error
        sum_absolute_errors += absolute_error    
          
    # We divide the sum of absolute errors by the length of y_true to compute the MAPE score
    mape = (sum_absolute_errors/len(y_true))
            
    return mape
```
```
# Using custom function
print('\nMean Absolute Percentage Error\n')
round(custom_mape(y_true, y_pred), 3)
```
```
Mean Absolute Percentage Error
0.034
```
This result tell that predictions deviate from the actual values by an average of 38%.

### Coefficient of Determinations (Rˆ2)
Also refered to as R-Squared is a measure that tell how well a regression model fits the actual data.
It quantifies the degree of which the variance in the dependent variable is predictable from the independent variables.

Values closer to 1.0 indicates a better model.

```
def custom_rsquared(y_true, y_pred):
    
    # Obtaining the mean of actual values
    mean_ytrue = sum(y_true) / len(y_true)
    
    # Obtaining the sum of the squared differences between actual and predicted valyes
    sum_of_squared_residuals = 0
    for true, predicted in zip (y_true, y_pred):
        sum_of_squared_residuals += (true - predicted) ** 2
    
    # Obtaining the total sum of squares
    total_sum_of_squares = 0
    for true in y_true:
        total_sum_of_squares += (true - mean_ytrue) ** 2
        
    # Computing the R-Squared score
    r_squared_score = 1 - (sum_of_squared_residuals / total_sum_of_squares)
        
    return r_squared_score
```
```
# Using custom function
print('\nCoefficient of Determination (R²)\n')
round(custom_rsquared(y_true, y_pred), 3)
```
```
Coefficient of Determination (R²)
0.98
```

A Rˆ2 score of 0.98 indicates a high level of correlation between ```y_true```and ```y_pred```and it suggests that the model could explaing about 98% of the variance in the observed data.
