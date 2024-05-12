Project Components:

**SCRAPING COMMENTS**

1.	Code Structure:

•	The project is divided into three sections, each targeting a specific Reddit post.
•	For each post, a URL is provided, and the Selenium web driver is used to access the post's page.
•	The script then locates and extracts comments using XPath.
2.	Libraries Used:

•	Selenium: Used for automating web browser interaction. The code utilizes the webdriver module to control the Chrome browser and the By module to locate elements on the web page.
•	time: Used for introducing delays in the script to ensure that the web page has fully loaded before attempting to scrape data.


Scraping Execution:

Post 1: The Duality of Japan
•	URL: The Duality of Japan
•	The script initializes a Chrome web driver, navigates to the provided URL, waits for the page to load, and then extracts comments using XPath.
•	Extracted comments are printed with an associated index.

Post 2: Debate Discussion Thread Open
•	URL: Debate Discussion Thread Open
•	Similar to Post 1, the script initializes a new web driver, navigates to the second Reddit post, waits for the page to load, and extracts comments using XPath.

Post 3: A Prison in El Salvador - A Country Once Known For...
•	URL: A Prison in El Salvador - A country once known for...
•	The script repeats the process for the third Reddit post, extracting comments with associated indices.
•	
Data Storage:
•	Extracted comments are stored in a list (lis) for each post, facilitating further analysis or storage in an external data structure or file.
Conclusion of scraping :

The project successfully achieves the goal of scraping comments from three different Reddit posts. However, it's crucial to note that web scraping comes with ethical considerations and should be performed in compliance with the terms of service of the website being scraped. Additionally, the script may need adjustments based on changes to the Reddit website structure.


**Text Preprocessing Pipeline
**

Objective:

The objective of this project is to preprocess text data from a CSV file containing comments. The preprocessing steps include converting text to lowercase, removing punctuation, stopwords, expanding contractions, removing emojis, and tokenizing the comments. The Python code utilizes the NLTK and Contractions libraries.

Processing Components:

1.	Importing Libraries and Installing Dependencies:

•	nltk and contractions libraries are imported.
•	The stopwords module and various classes/functions from the nltk library are downloaded.
•	
2.	Loading the Dataset:
•	The dataset is loaded from a CSV file named "Comments3.csv" into a Pandas DataFrame (df).
•	An unnamed column is dropped from the DataFrame for clarity.

3.	Lowercasing:
•	A custom function lower is defined to convert the 'Comments' column to lowercase using the str.lower() method.



4.	Removing Punctuation:
•	A custom function Rem_punct is defined to remove punctuation from the 'Comments' column using the str.replace() method.

5.	Removing Stopwords:
•	A custom function remove_stopwords is defined to remove stopwords from the 'Comments' column. Stopwords are common words that often do not contribute significant meaning to the text.

6.	Expanding Contractions:
•	A custom function exp is defined to expand contractions in the 'Comments' column using the contractions.fix() method.

7.	Removing Emojis:
•	A custom function deEmojify is defined to remove emojis from the 'Comments' column. This is achieved by encoding the text to ASCII and then decoding it.

8.	Tokenization:
•	The NLTK word_tokenize function is applied to tokenize the 'Comments' column, creating a new column named 'Tokenize_Comment'.

Conclusion:
•	The preprocessing pipeline is successfully executed, transforming the raw text data into a cleaner format suitable for further analysis or modeling.
•	The steps undertaken aim to enhance the quality of the text data by eliminating noise, irrelevant information, and standardizing the format.

Recommendations:
•	Consider additional text preprocessing techniques based on the specific requirements of the downstream tasks (e.g., sentiment analysis, topic modeling).
•	Regularly update the NLTK library and other dependencies to ensure compatibility with the latest features and improvements.
•	Document the preprocessing steps thoroughly for future reference.

Note: This report provides an overview of the text preprocessing pipeline and its individual steps, highlighting the purpose and impact of each operation on the dataset.

**MODELLING
**
Objective:

The objective of this project is to build a text classification model to predict whether a comment contains offensive language or not. The Python code utilizes various machine learning and deep learning techniques, including Naive Bayes, Support Vector Machines (SVM), and a Convolutional Neural Network (CNN).

Modelling Components:

1.	Importing Libraries:

•	Libraries such as pandas, numpy, re, CountVectorizer from scikit-learn, and modules from TensorFlow are imported.

2.	Loading the Dataset:

•	The dataset is loaded from a CSV file (presumably "Cleaned_Comments.csv") into a Pandas DataFrame (df).
•	The head of the DataFrame is printed for a quick overview.

3.	Data Inspection:

•	The data type of the 'Tokenize_Comment' column is checked. If it is not a string, it is converted to string type to ensure consistency.
•	
4.	Labeling Comments:
•	A list of "bad words" is defined, and a custom function comment_label is created to label each comment as 1 (offensive) or 0 (not offensive) based on the presence of any of the specified bad words.
•	The labels are added as a new column 'Labels' in the DataFrame.
•	
5.	Feature Extraction - Count Vectorization:

•	The 'Comments' column is vectorized using the CountVectorizer from scikit-learn to create a matrix of token counts.
•	The resulting feature matrix (X) and labels (y) are split into training and testing sets using the train_test_split function.

6.	Naive Bayes Classification:

•	A Naive Bayes classifier (MultinomialNB) is implemented using scikit-learn.
•	The model is trained on the training set (X_train, y_train) and evaluated on the test set (X_test, y_test).
•	The classification report is generated to assess the model's performance.

7.	Support Vector Machines (SVM) Classification:

•	A Support Vector Machines (SVM) classifier is implemented using scikit-learn.
•	Similar to Naive Bayes, the SVM model is trained, evaluated, and its performance is assessed using the classification report.

8.	Convolutional Neural Network (CNN) Classification:

•	A deep learning model, specifically a Convolutional Neural Network (CNN), is implemented using TensorFlow and Keras.
•	The 'Comments' column is preprocessed using tokenization and padding to ensure consistent input dimensions.
•	The model architecture includes an Embedding layer, Conv1D layer, GlobalMaxPooling1D layer, and Dense layer.
•	The model is trained on the training set and evaluated on the test set. The classification report is generated.

Conclusion:

•	The project successfully implements three different models for text classification: Naive Bayes, Support Vector Machines, and a Convolutional Neural Network.
•	The choice of models allows for a comparison of their performances on the specific task of identifying offensive language.
•	The classification report for each model provides insights into precision, recall, and F1-score, aiding in the assessment of model efficacy.

Recommendations:

•	Experiment with hyperparameter tuning for the SVM and CNN models to optimize performance.
•	Consider additional preprocessing steps or feature engineering to improve model accuracy.
•	Regularly update libraries and dependencies for access to the latest features and improvements.



**Machine Learning Model Evaluation Report:
**
Naive Bayes vs. Support Vector Machines (SVM)

1.	Naive Bayes Model:

•	Training:Utilized the Multinomial Naive Bayes classifier.
•	Achieved an accuracy of 95% on the test set.
•	Demonstrated good performance in predicting non-toxic comments (class 0) with a precision of 98%, recall of 97%, and F1-score of 98%.
•	Identified toxic comments (class 1) less effectively with lower precision (29%), recall (36%), and F1-score (32%).
•	Testing with a Sample Comment:Used a sample comment: "This is a sample comment! It contains offensive language."
•	Predicted as non-toxic (class 0) by the Naive Bayes model.



2.	Support Vector Machines (SVM) Model:

•	Training: Implemented the Support Vector Machines (SVM) classifier.
•	Achieved a higher accuracy of 97% on the test set compared to Naive Bayes.
•	Exhibited excellent performance in predicting non-toxic comments (class 0) with a precision of 97%, recall of 100%, and F1-score of 99%.
•	Struggled in identifying toxic comments (class 1) with a precision of 100%, recall of 9%, and F1-score of 17%.
•	Testing with a Sample Comment:Used a sample toxic comment: "fuck uou."
•	Predicted as toxic (class 1) by the SVM model.






Comparison:

•	Naive Bayes vs. SVM
•	Accuracy:Naive Bayes: 95% SVM: 97%
•	Non-toxic Class (0):Naive Bayes: Precision 98%, Recall 97%, F1-score 98%
•	SVM: Precision 97%, Recall 100%, F1-score 99%
•	Toxic Class (1):Naive Bayes: Precision 29%, Recall 36%, F1-score 32%
•	SVM: Precision 100%, Recall 9%, F1-score 17%
•	Observations:SVM outperforms Naive Bayes in overall accuracy.
•	Naive Bayes excels in predicting non-toxic comments, while SVM has superior precision and recall for toxic comments.
•	Both models have limitations in accurately identifying toxic comments, with Naive Bayes having lower recall and SVM struggling with precision.


Conclusion:

•	The SVM model demonstrates superior overall performance compared to Naive Bayes, achieving higher accuracy and better handling of non-toxic comments.
•	The choice between models may depend on specific project requirements, where precision or recall may be of greater importance.
•	Both models exhibit limitations in accurately identifying toxic comments, indicating the need for further model refinement or alternative approaches.
•	
Recommendations:
•	Consider experimenting with hyperparameter tuning for both models to optimize performance.
•	Explore additional features or advanced text representation techniques to improve model accuracy.
•	Monitor and address model limitations and biases, especially in the context of identifying toxic language.
•	
Note: This report provides an evaluation of the Naive Bayes and SVM models, comparing their performances on a text classification task. The assessment includes accuracy, precision, recall, and F1-score metrics for both non-toxic and toxic comment classes.
Model Evaluation Report: LSTM vs. CNN


1.	LSTM Model:

•	Training:Utilized the Long Short-Term Memory (LSTM) architecture for sequential data.
•	Achieved an accuracy of 96.97% on the test set.
•	Demonstrated a decreasing loss function and increasing accuracy across epochs.
•	Testing with a Sample Comment:Used a sample toxic comment: "This is an offensive and toxic comment."
•	Predicted as non-toxic (class 0) by the LSTM model.

2.	CNN Model:

•	Training:Implemented a Convolutional Neural Network (CNN) architecture.
•	Achieved a higher accuracy of 99.72% on the test set compared to LSTM.
•	Showed a decreasing loss function and increasing accuracy across epochs.
•	Testing with a Sample Comment:Used the same sample toxic comment as for LSTM.
•	Predicted as non-toxic (class 0) by the CNN model.

Comparison:
•	LSTM vs. CNN:Accuracy:LSTM: 96.97%
•	CNN: 99.72%
•	Observations:The CNN model outperforms the LSTM model in overall accuracy, demonstrating higher predictive performance.
•	Both models showed a decreasing loss function and increasing accuracy during training, indicating effective learning.
•	Interestingly, both models predict the sample toxic comment as non-toxic (class 0), highlighting potential limitations in handling specific instances of toxicity.

Conclusion:
•	The CNN model exhibits superior overall performance compared to the LSTM model, achieving higher accuracy on the test set.
•	The choice between LSTM and CNN may depend on the specific characteristics of the text data and the task requirements.
•	Both models, however, demonstrate limitations in correctly identifying toxicity in the provided sample comment.

Recommendations:

•	Experiment with hyperparameter tuning for both models to optimize performance.
•	Explore additional model architectures or advanced techniques to enhance the models' ability to identify toxic language.
•	Conduct further analysis on misclassified instances to understand model limitations and potential areas for improvement.

Note: This report provides an evaluation of the LSTM and CNN models, comparing their performances on a text classification task. The assessment includes accuracy metrics and observations on the training process. Additionally, the report highlights a limitation in both models in correctly predicting toxicity for a specific sample comment.






