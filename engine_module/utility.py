
class Utils:

    def email_mapper(self, email_serie):
        coded_dict = dict()
        cter = 1
        email_encoded = []

        for val in email_serie:
            if val not in coded_dict:
                coded_dict[val] = cter
                cter += 1

            email_encoded.append(coded_dict[val])
        return email_encoded

    def create_user_item_matrix(self, interactions_df):
        '''
        INPUT:
        df - pandas dataframe with article_id, title, user_id columns

        OUTPUT:
        user_item - user item matrix

        Description:
        Return a matrix with user ids as rows and article ids on the columns with 1 values where a user interacted with
        an article and a 0 otherwise
        '''
        # we create some sort of a pivot table using user_id and article_id as indexes and columns, and count as a value
        user_item = interactions_df.groupby(['user_id', 'article_id'])['user_id'].count().unstack()
        # we replace every numeric value by a 1 and every NaN by a 0
        user_item = user_item.notnull()
        user_item = user_item.astype('int')

        return user_item  # return the user_item matrix

    def get_article_names(self, article_ids, title_df):
        '''
        INPUT:
        article_ids - (list) a list of article ids
        title_df - (pandas dataframe) df containing the reference of the article id and title

        OUTPUT:
        article_names - (list) a list of article names associated with the list of article ids
                        (this is identified by the title column)
        '''
        article_names = []

        for article_id in article_ids:
            # we convert the input str value to float to match the type in the df
            article_id = float(article_id)
            article_name = title_df[title_df['article_id'] == article_id]['title'].values[0]
            article_names.append(article_name)

        return article_names  # Return the article names associated with list of article ids

    def remove_NaN(data_serie, value_to_fill):
        data_serie.fillna(value_to_fill, inplace=True)
        print('NaN successfully removed from the data serie!')
