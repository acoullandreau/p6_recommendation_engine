==================================================
IBM Recommendation Engine - Python module
==================================================


-----------------------
Purpose of this module
-----------------------

Implemented as part of the UDACITY Data Scientist Nanodegree, the purpose of this module is to allow the computation of recommendations of relevant to any user, known or unknown, and articles, from the IMB Watson Studio platform.

In this subrepository, you will find:

- the main.py script, in which some examples are implemented - this script could be adapter depending on the use to make of it
- the classfile.py containing all the recommenders classes
- the utility.py file containing support functions
- the data sourced used in the context of the
- this readme file


-----------------------
Installation requisites
-----------------------

The code is written in Python 3 (v3.6).

The following libraries were used extensively in the code:

- numpy 1.16.4
- pandas 0.25.0
- sklearn 0.21.2
- nltk 3.4.4
- plotly 4.1.1


--------------------
How to use this code
--------------------

The main script does not take any argument: simply run

.. code:: python

 python main.py

.. code:: python

The main.py script takes care of:
- loading the data source
- cleaning the data source
- creating an instance of RecommenderEngine

The instance of RecommenderEngine has the following properties:
- interactions_df, the dataframe containg all the interactions between users and articles
- article_content_df, the dataframe containg the details about each article
- user_item, a matrix of users (row) and articles (column) interactions (value 0 if no interaction and 1 otherwise)
- rank_recommender, an instance of the RankRecommender class, that recommends articles based on their total number of interactions
- user_recommender, an instance of the UserUserRecommender class, that recommends articles based on which articles the similar users interacted with
- content_recommender, an instance of the ContentRecommender class, that recommends articles based on their content similarity
- SVD_recommender, an instance of the SVDRecommender class, that recommends articles using SVD, i.e the prediction of the interactions of a user with an article

The RecommenderEngine.make_recommendations() method will take care of deciding which recommendation technique to use depending on the target of the recommendation list:
- if we want to get similar articles to a given article -> **ContentRecommender** for articles will provide the recommendations
- if we want to get recommendations to an unknown user -> **RankRecommender** will provide the recommendations
- if we want to get recommendations to a user who interacted with less than 5 articles -> **SVDRecommender** will provide the recommendations
- if we want to get similar articles to a user who interacted with more than 5 articles -> **ContentRecommender** for users and **UserUserRecommender** will provide the recommendations 

In the last scenario, we actually get 5 recommendations from UserUserRecommender and 5 recommendations from ContentRecommender for users. We then return the list of recommendations ranked per number of interactions. 

By default, all recommendations are provided as a list of titles, and a list of 10 titles is returned. 

Feel free to reuse any piece of code!


-------------------------------------------
Possible improvements
-------------------------------------------

Here is a list of possible improvements that could be made on this module:
- proper tests should be written to ensure that any changes on the code doesn't affect the output (automated tests to be constructed)
- currently the data source is not uniform (some articles are in just one of the two source files), and having csv files is not dynamic enough -> we could work with two DB tables, one with only article_id and user_id (interactions), and one containing the list of all articles with their details
- depending on the integration we want for this module, the main.py script should be adapted


