# Evaluation Metrics for Regression Models

## MAE: Mean Absolute Error

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
On average, the prediction is wrong by +_ 1.155 units. So if a data point is equal to 100, is expected the predictions ouput by the model for this specific data point to fluctuate from 98.845 to 101.555.
