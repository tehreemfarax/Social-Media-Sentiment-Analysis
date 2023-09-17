# Social-Media-Sentiment-Analysis

STEPS

**Data Exploration and Preprocessing:** The code begins by importing necessary libraries and loading the training and testing datasets. It conducts initial data exploration by displaying the shape and first few rows of the training data, checking for missing values, and visualizing aspects such as word length distribution and label counts. Text preprocessing functions are defined to clean and process the tweet text by removing special characters, converting to lowercase, and tokenizing.

**Feature Engineering and Model Training:** The code proceeds to perform feature engineering by utilizing the CountVectorizer and TfidfTransformer from Scikit-learn to convert the processed text into numerical features suitable for machine learning. It splits the data into training and testing sets, applies the chosen models (Random Forest, XGBoost, and SVM), and evaluates their performance using metrics like F1-score and confusion matrices. The results are stored and displayed.

**Grid Search and Hyperparameter Tuning:** The code demonstrates the use of GridSearchCV to perform hyperparameter tuning for a Multinomial Naive Bayes classifier. It explores different n-gram ranges, TF-IDF settings, and alpha values to find the optimal configuration for text classification. Finally, the code prints a classification report with detailed performance metrics for the tuned model. This section allows for the fine-tuning of the model's parameters to achieve better classification results.
