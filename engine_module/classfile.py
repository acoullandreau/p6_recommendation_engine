
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity

from utility import Utils

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


class SVDRecommender:

    def __init__(self, interactions_df, user_item):
        self.interactions_df = interactions_df
        self.user_item = user_item
        self.user_item_train = self.create_test_and_train_user_item()[0]
        self.user_item_test = self.create_test_and_train_user_item()[1]
        self.latent_features = 100
        self.s_matrix = self.get_SVD_matrices()[0]
        self.u_matrix = self.get_SVD_matrices()[1]
        self.vt_matrix = self.get_SVD_matrices()[2]

    def create_test_and_train_user_item(self):
        '''
        INPUT:
        df_train - training dataframe
        df_test - test dataframe

        OUTPUT:
        user_item_train - a user-item matrix of the training dataframe 
                          (unique users for each row and unique articles for each column)
        user_item_test - a user-item matrix of the testing dataframe 
                        (unique users for each row and unique articles for each column)

        '''
        num_interactions = len(self.interactions_df)
        len_train = int(70*num_interactions/100)  # 70% of the df for train
        len_test = num_interactions - len_train  # 30% of the df for test
        df_train = self.interactions_df.head(len_train)
        df_test = self.interactions_df.tail(len_test)

        # we reuse the create_user_item_matrix we defined earlier
        user_item_train = Utils.create_user_item_matrix(df_train)
        user_item_test = Utils.create_user_item_matrix(df_test)

        return (user_item_train, user_item_test)

    def get_SVD_matrices(self):
        # Perform SVD on the User-Item Matrix Here
        u, s, vt = np.linalg.svd(self.user_item)
        s_new, u_new, vt_new = np.diag(s[:self.latent_features]), u[:, :self.latent_features], vt[:self.latent_features, :]

        return (s_new, u_new, vt_new)

    def make_SVD_recommendations(self, user_id, num_recommendations=10):
        preds = np.around(np.dot(np.dot(self.u_matrix, self.s_matrix), self.vt_matrix))
        users = self.user_item.index
        user_idx = np.where(users == user_id)[0][0]
        articles_idx = preds.argsort()[-num_recommendations:][::-1]

        rec_ids = self.user_item.columns[articles_idx]
        recommended_articles = Utils.get_movie_names(rec_ids, self.interactions_df)

        return recommended_articles


class RankRecommender:

    def __init__(self, interactions_df):
        self.interactions_df = interactions_df
        self.article_ids = self.interactions_df['article_id'].unique()
        self.top_articles_df = self.get_top_articles_df()
        self.recommendations_names = self.get_top_articles()
        self.recommendations_ids = None

    def get_top_articles_df(self):
        '''
        INPUT:
        self.article_ids - (list) list of articles to use as a base for recommendations
        self.interactions_df - (pandas dataframe) df of all article-user interactions

        OUTPUT:
        self.top_articles_df - (dataframe) A df of ranked articles per number of interactions

        '''

        articles_dict = {}

        for article in self.article_ids:
            article = float(article)
            interact = len(self.interactions_df[self.interactions_df['article_id'] == article])
            article_title = self.interactions_df[self.interactions_df['article_id'] == article]['title'].values[0]
            articles_dict[article] = {'num_interactions': interact, 'title': article_title}

        top_articles_df = pd.DataFrame.from_dict(articles_dict, orient='index')
        top_articles_df = top_articles_df.sort_values(by='num_interactions', ascending=False)

        self.top_articles_df = top_articles_df

    def get_top_articles(self, num_recommendations=10):
        '''
        INPUT:
        self.num_recommendations - (int) the number of top articles to return
        self.top_articles_df - (dataframe) A df of ranked articles per number of interactions

        OUTPUT:
        self.recommendations_names - (list) A list of the top 'n' article titles

        '''

        # we select only the num_recommendations top articles from the sorted df
        self.recommendations_names = self.top_articles_df.head(num_recommendations)['title'].values

    def get_top_article_ids(self, num_recommendations=10):
        '''
        INPUT:
        self.num_recommendations - (int) the number of top articles to return
        self.top_articles_df - (dataframe) A df of ranked articles per number of interactions

        OUTPUT:
        self.recommendations_names - (list) A list of the top 'n' article ids (str format)

        '''

        # we select only the num_recommendations top articles from the sorted df
        top_articles = self.top_articles_df.head(num_recommendations).index.values
        top_articles = [str(i) for i in top_articles]
        self.recommendations_ids = top_articles


class UserUserRecommender:

    def __init__(self, interactions_df, user_item):
        self.interactions_df = interactions_df
        self.user_item = user_item
        self.similarity_matrix = self.compute_similarity()

    def compute_similarity(self):
        # compute similarity of each user to any other user
        dot_prod = self.user_item.dot(self.user_item.transpose())
        self.similarity_matrix = dot_prod

    def find_similar_users(self, user_id):
        '''
        INPUT:
        user_id - (int) a user_id
        self.user_item - (pandas dataframe) matrix of users by articles:
                    1's when a user has interacted with an article, 0 otherwise

        OUTPUT:
        similar_users - (list) an ordered list where the closest users (largest dot product users)
                        are listed first

        '''

        # get the list of neighbors in descending order
        user_row = np.where(self.similarity_matrix.index == user_id)[0][0]
        neighbors = self.similarity_matrix.iloc[user_row].sort_values(ascending=False)

        # create list of just the ids
        most_similar_users = neighbors.index.values

        # remove the own user's id
        most_similar_users = np.delete(most_similar_users, np.argwhere(most_similar_users == user_id))

        return most_similar_users  # returns a list of the users in order from most to least similar

    def get_user_articles(self, user_id):
        '''
        INPUT:
        user_id - (int) a user id
        user_item - (pandas dataframe) matrix of users by articles:
                    1's when a user has interacted with an article, 0 otherwise

        OUTPUT:
        article_ids - (list) a list of the article ids seen by the user
        article_names - (list) a list of article names associated with the list of article ids
                        (this is identified by the doc_full_name column in df_content)

        Description:
        Provides a list of the article_ids and article titles that have been seen by a user
        '''

        user_row = np.where(self.user_item.index == user_id)[0][0]
        user_articles = np.where(self.user_item.iloc[user_row] == 1)[0]
        article_ids = []

        for article in user_articles:
            article_id = self.user_item.iloc[:, article].name
            article_ids.append(str(article_id))  #to match the expected str type as output

        article_names = Utils.get_article_names(article_ids, self.interactions_df)

        return article_ids, article_names  # return the ids and names

    def get_top_sorted_users(self, user_id):
        '''
        INPUT:
        user_id - (int)
        df - (pandas dataframe) df as defined at the top of the notebook
        user_item - (pandas dataframe) matrix of users by articles:
                1's when a user has interacted with an article, 0 otherwise

        OUTPUT:
        neighbors_df - (pandas dataframe) a dataframe with:
                        neighbor_id - is a neighbor user_id
                        similarity - measure of the similarity of each user to the provided user_id
                        num_interactions - the number of articles viewed by the user - if a u

        Other Details - sort the neighbors_df by the similarity and then by number of interactions where
                        highest of each is higher in the dataframe

        '''
        neighbors_dict = {}
        neighbors_list = self.find_similar_users(user_id)

        for neighbor in neighbors_list:
            # we use the dot_prod df already build earlier - can be built as part of find_similar_users
            similarity_score = self.similarity_matrix.loc[user_id, neighbor]
            num_interactions = len(self.interactions_df[self.interactions_df['user_id'] == neighbor])
            neighbors_dict[neighbor] = {'similarity': similarity_score, 'num_interactions': num_interactions}

        neighbors_df = pd.DataFrame.from_dict(neighbors_dict, orient='index')
        neighbors_df = neighbors_df.sort_values(by=['similarity', 'num_interactions'], ascending=False)

        return neighbors_df  # Return the dataframe specified in the doc_string

    def make_user_user_recommendations(self, user_id, num_recommendations=10):
        '''
        INPUT:
        user_id - (int) a user id
        num_recommendations - (int) the number of recommendations you want for the user

        OUTPUT:
        recs - (list) a list of recommendations for the user by article id
        rec_names - (list) a list of recommendations for the user by article title

        Description:
        Loops through the users based on closeness to the input user_id
        For each user - finds articles the user hasn't seen before and provides them as recs
        Does this until m recommendations are found

        Notes:
        * Choose the users that have the most total article interactions
        before choosing those with fewer article interactions.

        * Choose articles with the articles with the most total interactions
        before choosing those with fewer total interactions.

        '''

        recs = []

        neighbors_df = self.get_top_sorted_users(user_id)
        user_articles_id, user_articles_names = self.get_user_articles(user_id)

        for neighbor in neighbors_df.index:
            neighbor_articles_id, neighbor_articles_names = self.get_user_articles(neighbor)
            sorted_neighbor_article_ids = self.get_top_articles_df(neighbor_articles_id)
            sorted_neighbor_article_ids = sorted_neighbor_article_ids.index.values
            article_not_read = np.setdiff1d(sorted_neighbor_article_ids, user_articles_id, assume_unique=True)
            article_not_read = [str(i) for i in article_not_read]
            recs = np.unique(np.concatenate([article_not_read, recs], axis=0))

            if len(recs) >= num_recommendations:
                break

        if len(recs) >= num_recommendations:
            recs = recs[:10]

        recommended_articles = self.get_article_names(recs)

        return recommended_articles


class ContentRecommender:

    def __init__(self, interactions_df, article_content_df, user_item):
        self.stop_words = stopwords.words("english")
        self.lemmatizer = WordNetLemmatizer()
        self.interactions_df = interactions_df
        self.article_content_df = article_content_df
        self.content_analysis_target = self.article_content_df['doc_description']
        self.user_item = user_item
        self.content_similarity_matrix = self.compute_content_similarity_matrix()

    def NLP_processing(self, text):
        # initialize count vectorizer object
        vect = CountVectorizer(lowercase=False, tokenizer=self.tokenize)
        # get counts of each token (word) in text data
        X = vect.fit_transform(self.content_analysis_target)

        # initialize tf-idf transformer object
        transformer = TfidfTransformer(smooth_idf=False)
        # use counts from count vectorizer results to compute tf-idf values
        tfidf = transformer.fit_transform(X)

        return tfidf

    def compute_content_similarity_matrix(self):
        tfidf_matrix = self.NLP_processing(self.content_analysis_target)

        cosine = cosine_similarity(tfidf_matrix, tfidf_matrix)
        self.content_similarity_matrix = cosine

    def tokenize(self, text):
        """
            Data cleaning and tranformation function, that parses the data and
            outputs a simplified data content.
            Parsing includes:
            - removal of the URLs
            - removal of the punctuation
            - tokenization of the text
            - lemmatization of the tokenized text (words)

            Inputs:
                text - the data to transform
            Outputs:
                words - the words contained in the data after parsing
        """

        # we cast the text to string
        text = str(text)

        # we convert the text to lower case
        text = text.lower()

        # we remove any url contained in the text

        url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        url_in_text = re.findall(url_regex, text)
        for url in url_in_text:
            text = text.replace(url, "urlplaceholder")

        # we remove the punctuation
        text = re.sub(r"[^a-z\s]", " ", text)

        # we tokenize the text
        words = word_tokenize(text)

        # we lemmatize  and remove the stop words
        words = [self.lemmatizer.lemmatize(word) for word in words if word not in self.stopwords.words('english')]

        return words

    def find_similar_articles(self, article_id):
        '''
        INPUT
        article_id - an article_id
        self.content_similarity_matrix - the cosine similarity matrix
        self.article_content_df - the df containing details about the articles

        OUTPUT
        similar_articles - a list of the most similar articles by ids, sorted
        per similarity, based on the content_similarity_matrix
        '''

        # find the row of each article id
        article_row = self.article_content_df[self.article_content_df['article_id'] == article_id].index[0]

        # find the most similar article indices
        similar_articles = np.argsort(self.content_similarity_matrix[article_row])[::-1]
        similar_articles = np.delete(similar_articles, np.argwhere(similar_articles == article_id))

        return similar_articles

    def get_user_articles(self, user_id):
        '''
        INPUT:
        user_id - (int) a user id
        user_item - (pandas dataframe) matrix of users by articles:
                    1's when a user has interacted with an article, 0 otherwise

        OUTPUT:
        article_ids - (list) a list of the article ids seen by the user
        article_names - (list) a list of article names associated with the list of article ids
                        (this is identified by the doc_full_name column in df_content)

        Description:
        Provides a list of the article_ids and article titles that have been seen by a user
        '''

        user_row = np.where(self.user_item.index == user_id)[0][0]
        user_articles = np.where(self.user_item.iloc[user_row] == 1)[0]
        article_ids = []

        for article in user_articles:
            article_id = self.user_item.iloc[:, article].name
            article_ids.append(str(article_id))  # to match the expected str type as output

        article_names = Utils.get_article_names(article_ids, self.interactions_df)

        return article_ids, article_names  # return the ids and names

    def make_content_article_recommendations(self, _id, num_recommendations=10):
        '''
        INPUT:
        _id, the id of the article we want similar articles for
        self.content_similarity_matrix, the similarity matrix of the articles, by default cosine matrix computed separately
        self.interactions_df, the dataframe with the interactions of users with articles
        self.article_content_df - the df containing details about the articles
        num_recommendations, the number of recommendations expected as an output, by default 10

        OUTPUT:
        recommended_articles, a list of similar articles, given by name
        '''

        recommended_articles = self.find_similar_articles(_id)
        recommended_articles = recommended_articles[:num_recommendations]
        recommended_articles = Utils.get_article_names(recommended_articles, self.article_content_df)

        return recommended_articles

    def make_content_user_recommendations(self, _id, num_recommendations=10):
        '''
        INPUT:
        _id, the id of the user we want recommended articles for
        self.content_similarity_matrix, the similarity matrix of the articles, by default cosine matrix computed separately
        self.interactions_df, the dataframe with the interactions of users with articles
        self.article_content_df - the df containing details about the articles
        num_recommendations, the number of recommendations expected as an output, by default 10

        OUTPUT:
        recommended_articles, a list of recommended articles, given by name
        '''

        # get the articles a user read
        user_articles_id, user_articles_names = self.get_user_articles(_id)

        # filter out the articles that are not in the df of article details
        user_articles_id = [float(i) for i in user_articles_id]
        user_articles = self.article_content_df[self.article_content_df['article_id'].isin(user_articles_id)]['article_id'].values

        # sort the articles_id per number of interactions
        user_article_inter_dict = {}
        for article in user_articles:
            interact = len(self.interactions_df[(self.interactions_df['user_id'] == _id) & (self.interactions_df['article_id'] == article)])
            article_title = self.interactions_df[self.interactions_df['article_id'] == article]['title'].values[0]
            user_article_inter_dict[article] = {'num_interactions': interact, 'title': article_title}

        top_user_articles_df = pd.DataFrame.from_dict(user_article_inter_dict, orient='index')
        top_user_articles_df = top_user_articles_df.sort_values(by='num_interactions', ascending=False)

        # find similar articles in order
        recommended_articles = []
        for article in top_user_articles_df.index:
            articles_sim = self.find_similar_articles(article)
            unread_articles = np.setdiff1d(articles_sim, top_user_articles_df.index, assume_unique=True)
            for unread_article in unread_articles:
                if unread_article not in recommended_articles:
                    recommended_articles.append(unread_article)

            if len(recommended_articles) > num_recommendations:
                break

        recommended_articles = recommended_articles[:num_recommendations]
        recommended_articles = Utils.get_article_names(recommended_articles, self.article_content_df)

        return recommended_articles


class RecommendationEngine:

    def __init__(self, interactions_df, article_content_df, user_item):
        self.interactions_df = interactions_df
        self.article_content_df = article_content_df
        self.user_item = user_item
        self.rank_recommender = RankRecommender(self.interactions_df)
        self.user_recommender = UserUserRecommender(self.interactions_df, self.user_item)
        self.content_recommender = ContentRecommender(self.interactions_df, self.user_item)
        self.SVD_recommender = SVDRecommender(self.interactions_df, self.article_content_df, self.user_item)

    def make_recommendations(self, _id, _id_type, num_recommendations=10):

        if _id_type == 'user':
            # check if user is known
            if _id in np.array(self.interactions_df['user_id']):
                # check if user had interactions with more than 5 articles
                user_num_interactions = len(self.interactions_df[(self.interactions_df['user_id'] == _id)]['article_id'].unique())
                if user_num_interactions < 5:
                    recommended_articles = SVDRecommender.make_SVD_recommendations(_id, num_recommendations)
                else:
                    user_like_num_recs = int(num_recommendations/2)
                    content_like_num_recs = num_recommendations - user_like_num_recs
                    # we create a recommendation list
                    user_like_articles = UserUserRecommender.make_user_user_recommendations(_id, num_recommendations=user_like_num_recs)
                    content_like_articles = ContentRecommender.make_content_user_recommendations(_id, num_recommendations=content_like_num_recs)
                    recommended_articles = user_like_articles + content_like_articles

                    # we sort the articles per number of interactions
                    top_articles_df = RankRecommender.top_articles_df
                    top_articles_ranked = top_articles_df[top_articles_df.index.isin(recommended_articles)]

                    recommended_articles = [article for _, article in sorted(zip(top_articles_ranked, recommended_articles))]

            else:
                # if user is unknown, we switch to ranked based recommendations
                recommended_articles = RankRecommender.get_top_articles(num_recommendations)

        elif _id_type == 'article':
            recommended_articles = self.content_recommender.make_content_article_recommendations(_id, num_recommendations)

        else:
            print('Please specify a valid id type: article or user.')

        return recommended_articles
