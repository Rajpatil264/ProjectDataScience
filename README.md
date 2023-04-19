# ProjectDataScience
This is a Python code that performs data analysis and predictive modeling on a banking dataset, using machine learning algorithms such as logistic regression and decision tree classifier.

The code starts by importing necessary libraries such as pandas, numpy, seaborn, and matplotlib.
Then, it loads the training and testing dataset from local directories using pandas.
It prints the columns and shape of the training and testing dataset, as well as their datatypes and first five rows.
It then performs data visualization using seaborn and matplotlib to gain insights into the dataset. This includes plotting bar plots, distribution plots, and cross-tabulation plots to analyze the relationship between different variables.
Next, it replaces the categorical variable 'subscribed' with 0s and 1s for easy modeling.
It calculates the correlation between the variables and plots a heatmap to visualize it.
It checks for missing values in the dataset.
It splits the dataset into training and validation sets and applies one-hot encoding to categorical variables in the dataset.
It trains and evaluates a logistic regression model and a decision tree model on the dataset.
Finally, it makes predictions on the test dataset using the decision tree model and saves the predictions in a CSV file.
