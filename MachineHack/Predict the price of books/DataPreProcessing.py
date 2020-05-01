import pandas as pd
import numpy as np
from dateutil.parser import parse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def is_date(string, fuzzy=False):
    """
    Return whether the string can be interpreted as a date.
    :param string: str, string to check for date
    :param fuzzy: bool, ignore unknown tokens in string if True
    """
    try:
        parse(string, fuzzy=fuzzy)
        return True

    except ValueError:
        return False


def book_language(book_edition):
    """
    Extract language of the book, from "Edition" field.
    :param book_edition: book edition
    :return: book language
    """

    book_lang = book_edition.split(',')[0]

    if book_lang in ['(Kannada)', '(French)', '(Spanish)', '(Chinese)', '(German)']:
        return book_lang
    else:
        return '(Unknown)'


def book_binding(book_edition):
    """
    Extract binding of the book, from "Edition" field.
    :param book_edition: book edition
    :return: book binding
    """

    if book_edition.split(',')[0] in ['(Kannada)', '(French)', '(Spanish)', '(Chinese)', '(German)']:
        return book_edition.split(',')[1]
    else:
        return book_edition.split(',')[0]


def book_type(book_edition):
    """
    Extract type of the book, from "Edition" field.
    :param book_edition: book edition
    :return: book type
    """

    first_field = book_edition.split('–')[1].split(',')[0].strip()
    if not (is_date(first_field) or first_field == 'Import'):
        return first_field
    else:
        return 'Unknown'


def book_import(book_edition):
    """
    Extract edition of the book, from "Edition" field.
    :param book_edition: book edition
    :return: book import flag
    """

    first_field = book_edition.split('–')[1].split(',')[0].strip()
    second_field = book_edition.split('–')[1].split(',')[1].strip() \
        if len(book_edition.split('–')[1].split(',')) > 1 else 'Import'

    if len(book_edition.split('–')[1].split(',')) > 1 and (first_field == 'Import' or second_field == 'Import'):
        return 'Y'
    else:
        return 'N'


def book_publishing_date(book_edition):
    """
    Extract publishing date of the book, from "Edition field.
    :param book_edition: book edition
    :return: book publishing date
    """

    field_length = len(book_edition.split('–')[1].split(','))

    if field_length == 1 and is_date(book_edition.split('–')[1].split(',')[0].strip()):
        return parse(book_edition.split('–')[1].split(',')[0].strip())
    elif field_length == 2 and is_date(book_edition.split('–')[1].split(',')[1].strip()):
        return parse(book_edition.split('–')[1].split(',')[1].strip())
    elif field_length == 3 and is_date(book_edition.split('–')[1].split(',')[2].strip()):
        return parse(book_edition.split('–')[1].split(',')[2].strip())
    else:
        return parse('0001')


# Set file paths for train and predict datasets
train_file = 'DataSet/Data_Train.xlsx'
predict_file = 'DataSet/Data_Test.xlsx'

# Extract train/predict data from spreadsheet into pandas dataframes
train_df = pd.read_excel(train_file)
predict_df = pd.read_excel(predict_file)

# Extract "Price" field from train_df into NumPy array
train_y = np.array([train_df['Price'].values]).T
train_df.drop(['Price'], inplace=True, axis=1)
print("train_y: {}".format(train_y.shape))

# Combine the train and predict dataframes
combined_df = train_df.append(predict_df, sort=False, ignore_index=True)

# Column encoding for "Title" and "Author" field
combined_df['title_enc'] = combined_df['Title'].factorize()[0]
combined_df['author_enc'] = combined_df['Author'].factorize()[0]

# Extract new features from "Edition" field
combined_df['book_language'] = combined_df['Edition'].map(book_language)
combined_df['book_binding'] = combined_df['Edition'].map(book_binding)
combined_df['book_type'] = combined_df['Edition'].map(book_type)
combined_df['book_import'] = combined_df['Edition'].map(book_import)
combined_df['book_publishing_date'] = combined_df['Edition'].map(book_publishing_date)

# Extract new features from "book_publishing_date" field
combined_df['pub_date_year'] = combined_df['book_publishing_date'].map(lambda x: pd.to_datetime(x).year)
combined_df['pub_date_quarter'] = combined_df['book_publishing_date'].map(lambda x: pd.to_datetime(x).quarter)
combined_df['pub_date_month'] = combined_df['book_publishing_date'].map(lambda x: pd.to_datetime(x).month)
combined_df['pub_date_week'] = combined_df['book_publishing_date'].map(lambda x: pd.to_datetime(x).week)
combined_df['pub_date_day_year'] = combined_df['book_publishing_date'].map(lambda x: pd.to_datetime(x).dayofyear)
combined_df['pub_date_day_month'] = combined_df['book_publishing_date'].map(lambda x: pd.to_datetime(x).day)
combined_df['pub_date_day_week'] = combined_df['book_publishing_date'].map(lambda x: pd.to_datetime(x).dayofweek)

# Column encoding for ['book_language','book_binding','book_type','book_import'] fields
combined_df['book_lang_enc'] = combined_df['book_language'].factorize()[0]
combined_df['book_bind_enc'] = combined_df['book_binding'].factorize()[0]
combined_df['book_typ_enc'] = combined_df['book_type'].factorize()[0]
combined_df['book_imp_enc'] = combined_df['book_import'].factorize()[0]

# Extract new features from "Reviews" field
combined_df['rating'] = combined_df['Reviews'].map(lambda x: x.split(' out of ')[0])
combined_df['max_rating'] = combined_df['Reviews'].map(lambda x: x.split(' out of ')[1].split('stars')[0].strip())
combined_df['rating_pct'] = (combined_df['rating'].astype(float) / combined_df['max_rating'].astype(float)) * 100

# Extract new features from "Ratings" field
combined_df['no_of_cust_reviews'] = combined_df['Ratings'].map(lambda x: x.split('customer review')[0].strip().
                                                               replace(",", ""))

# Column encoding for "Synopsis" field
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, ngram_range=(1, 1), stop_words='english', max_features=10000)
features = tfidf.fit_transform(combined_df.Synopsis).toarray()
features_df = pd.DataFrame(features, columns=tfidf.get_feature_names())
combined_df = pd.merge(combined_df, features_df, left_index=True, right_index=True)

# Column encoding for "Genre" and "BookCategory" fields
combined_df['genre_enc'] = combined_df['Genre'].factorize()[0]
combined_df['book_category_enc'] = combined_df['BookCategory'].factorize()[0]

# Drop redundant columns
combined_df.drop(['Title', 'Author', 'Edition', 'book_publishing_date', 'Reviews', 'max_rating', 'Ratings', 'Synopsis',
                  'Genre', 'BookCategory', 'book_language', 'book_binding', 'book_type', 'book_import'],
                 inplace=True, axis=1)

# Check if any column has NaN value in dataframe
print("Column with NaN value: {}".format(combined_df.columns[combined_df.isnull().any()].tolist()))

# Segregate combined_df into train/predict datasets
train_x = combined_df[:6237]
predict_x = combined_df[6237:]

# Scale the train/predict datasets
scaler = StandardScaler().fit(train_x)
train_x = scaler.transform(train_x)
predict_x = scaler.transform(predict_x)
print("train_x: {}".format(train_x.shape))
print("predict_x: {}".format(predict_x.shape))

# Scale train_y dataset
train_y = (train_y - train_y.mean()) / train_y.std()

# Split training dataset into train/validation/test datasets
X_train, X_val, Y_train, Y_val = train_test_split(train_x, train_y, test_size=0.1, random_state=1)
X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.06, random_state=1)

# Save the processed datasets into NPZ file
np.savez_compressed('DataSet/processed_dataset.npz',
                    X_train=X_train,
                    Y_train=Y_train,
                    X_val=X_val,
                    Y_val=Y_val,
                    X_test=X_test,
                    Y_test=Y_test,
                    X_predict=predict_x)
print("\nProcessed dataset has been stored at the path: DataSet/processed_dataset.npz")
