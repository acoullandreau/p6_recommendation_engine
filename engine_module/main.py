import pandas as pd

from classfile import RecommendationEngine
from utility import Utils

# import and cleandata sources
interactions_df = pd.read_csv('data/user-item-interactions.csv')
article_content_df = pd.read_csv('data/articles_community.csv')
del interactions_df['Unnamed: 0']
del article_content_df['Unnamed: 0']

email_encoded = Utils.email_mapper(interactions_df['email'])
del interactions_df['email']
interactions_df['user_id'] = email_encoded

# create a matrix of user-article interactions
user_item = Utils.create_user_item_matrix(interactions_df)

# create an instance of the Recommendation Engine that can be used for multiple situations
rec_engine = RecommendationEngine(interactions_df, article_content_df, user_item)

# test the code for a few situations (expected returned output of 10 article titles)
# recommendations for an article
_id_type = 'article'
_id = 10
recommended_articles = rec_engine.make_recommendations(_id, _id_type)
print('Test article')
print('The following articles are recommended based on your query for {} id {}:'.format(_id_type, _id))
print(recommended_articles)
print('\n')

# recommendations for an unknown user
_id_type = 'user'
_id = 5150
print('Unknown user')
recommended_articles = rec_engine.make_recommendations(_id, _id_type)
print('The following articles are recommended based on your query for {} id {}:'.format(_id_type, _id))
print(recommended_articles)
print('\n')

# recommendations for a user with interactions with less than 5 articles
_id_type = 'user'
_id = 2573
print('Test user with interactions with less than 5 articles')
recommended_articles = rec_engine.make_recommendations(_id, _id_type)
print('The following articles are recommended based on your query for {} id {}:'.format(_id_type, _id))
print(recommended_articles)
print('\n')

# recommendations for user with interactions with more than 5 articles
_id_type = 'user'
_id = 8
print('Test user with interactions with more than 5 articles')
recommended_articles = rec_engine.make_recommendations(_id, _id_type)
print('The following articles are recommended based on your query for {} id {}:'.format(_id_type, _id))
print(recommended_articles)
