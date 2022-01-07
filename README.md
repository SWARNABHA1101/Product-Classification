# Product-Classification
Multi-Class Text Classification for products based on their description with Machine Learning algorithms and Neural Networks (MLP, NLP,CNN, Distilbert).

## Product Categorization
### Multi-Class Text Classification of products based on their description
 
### General info

The goal of the project is product categorization based on their description with Machine Learning and Deep Learning (MLP, CNN, Distilbert) algorithms. Additionaly we have created Doc2vec and Word2vec models, Topic Modeling (with LDA analysis) and EDA analysis (data exploration, data aggregation and cleaning data).

The dataset comes from amazon product items with category labels.

### Motivation

The aim of the project is multi-class text classification to make-up products based on their description. Based on given text as an input, we have predicted what would be the category. We have five types of categories corresponding to different makeup products. In our analysis we used a different  methods for a feature extraction (such as Word2vec, Doc2vec) and various Machine Learning/Deep Lerning algorithms to get more accurate predictions and choose the most accurate one for our issue. 

### Project contains:
* Multi-class text classification with ML algorithms- ***Text_analysis.ipynb***
* Text classification with Distilbert model - ***Bert_products.ipynb***
* Text classification with MLP and Convolutional Neural Netwok (CNN) models - ***Text_nn.ipynb***
* Text classification with Doc2vec model -***Doc2vec.ipynb***
* Word2vec model - ***Word2vec.ipynb***
* LDA - Topic modeling - ***LDA_Topic_modeling.ipynb***
* EDA analysis - ***Products_analysis.ipynb***
* Python scripts to clean data and ML model - **clean_data.py, text_model.py**
* data, models - data and models used in the project.
### Workflow:
Data processing:
* Downsample to make a relatively balanced classes
* One-hot encode the labels
* Tokenize the texts then turn them into padded sequences
* Train test split the sequences and labels.

Construct NN:
* SpatialDropout1D performs variational dropout in NLP models.
* The next layer is the LSTM layer with 64 memory units.
* The output layer must create 7 output values, one for each class.
* Activation function is softmax for multi-class classification.
* Because it is a multi-class classification problem, categorical_crossentropy is used as the loss function.

Result:
After 10 epochs, on test set
Loss: 0.508
Accuracy: 0.832

<a href="url"><img src="https://github.com/JinghuiZhao/product-item-name-classification/blob/master/lstm_pred.png" height="800" width="500" ></a>
<a href="url"><img src="https://github.com/JinghuiZhao/product-item-name-classification/blob/master/lstm_pred2.png" height="800" width="400" ></a>
## Comments and thoughts:
* When it comes to creating features:
Women are the main consumers for beauty(cosmetics) and jewelery stuffs.
Men are more likely to purchase electronic devices and tools Teddoler/boys/girls are more
likely to purchase toys. If we have gender/ age group information, we can make features out of that and use
FeatureUnion from sklearn.pipeline to combine this feature with the vectorization of each
product name.
* For supervised model, since the model is trained on the labeled Amazon data, the kinds of classes should be more aglined with ecommerce product name dataset. And the data balance also matters.
* I can also try more advanced deep learning model like bi-directional LSTM.
###  How can you extract the additional information from the item names, such as the color, style, size, material, gender etc. if there is any?

<li> It would make sense if I extract the additional information under each class. Multileveled or hirearchical classifier would achieve better accuracy and save more time, since for each category, the adjective words can be really different. For example, in the most frequent fixed 2 grams, ‘Genuine Leather’ appears with ‘Belt’, ‘Bath’ appears with ‘Towel’, rather than ‘Genuine Leather’ appears with ‘towel’. </li>

``` CBOW and skip-gram ``` 
<li> Word2vec is one of the most popular technique to learn word embeddings using a two-layer neural network. Its input is a text corpus and its output is a set of vectors. There are two main training algorithms for word2vec, one is the continuous bag of words(CBOW), another is called skip-gram. These 2 will help us learn the similarity of words. </li>
<li> The major difference between these two methods is that CBOW is using context to predict a target word while skip-gram is using a word to predict a target context. Generally, the skip-gram method can have a better performance compared with CBOW method, for it can capture two semantics for a single word. For instance, it will have two vector representations for Apple, one for the company and another for the fruit. This is also the case in my test code. </li>

## Summary

We begin with data analysis and data pre-processing from our dataset. Then we have used a few combination of text representation such as BoW and TF-IDF and we have trained the word2vec and doc2vec models from our data. We have experimented with several Machine Learning algorithms: Logistic Regression, Linear SVM, Multinomial Naive Bayes, Random Forest, Gradient Boosting and MLP and Convolutional Neural Network (CNN) using different combinations of text representations and embeddings. We have also used a pretrained Distilbert model from Huggingface Transformers library to resolve our problem. We applied a transfer learning with Distilbert model. 

From our experiments we can see that the tested models give a overall high accuracy and similar results for our problem. The SVM (BOW +TF-IDF) model and MLP model give the best  accuracy of validation set. Logistic regression performed very well both with BOW +TF-IDF and Doc2vec and achieved similar accuracy as MLP. CNN with word embeddings also has a very comparable result (0.93) to MLP. Transfer learning with Distilbert model also gave a similar results to previous models. We achieved an accuracy on the test set equal to 93 %. That shows the extensive models are not gave a better results to our problem than simple Machine Learning models such as SVM. 


Model | Embeddings | Accuracy
------------ | ------------- | ------------- 
**CNN** | **Word embedding** | **0.93**
Distilbert| Distilbert tokenizer | 0.93
MLP| Word embedding  | 0.93
SVM | Doc2vec (DBOW)| 0.93
SVM| BOW +TF-IDF  | 0.93
Logistic Regression | Doc2vec (DBOW) | 0.91
Gradient Boosting | BOW +TF-IDF | 0.91
Logistic Regression | BOW +TF-IDF  | 0.91
Random Forest| BOW +TF-IDF | 0.91
Naive Bayes | BOW +TF-IDF | 0.90
Logistic Regression | Doc2vec (DM)  | 0.89


#### The project is created with:

* Python 3.6/3.8
* libraries: NLTK, gensim, Keras, TensorFlow, Hugging Face transformers, scikit-learn, pandas, numpy, seaborn, pyLDAvis.

#### Running the project:

* To run this project use Jupyter Notebook or Google Colab.
