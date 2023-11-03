import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
from src.nlp_modeler import NLPModeler


class DataTransformer: 

    """
    A class for loading, preprocessing, and vectorizing data.

    Methods:
        load_data(filename: str) -> pd.DataFrame:
            Load data from an Excel file and filter it based on specific criteria.

        preprocess_questions(df: pd.DataFrame) -> pd.DataFrame:
            Preprocess the questions in a DataFrame by lemmatizing, removing stopwords, and punctuation.

        vectorize(text_col: pd.Series, label: pd.Series,
                  ngram_range: tuple = (1, 2), min_df: int = 5) -> (pd.DataFrame, pd.Series):
            Vectorize the text column using CountVectorizer and return the feature matrix and labels.
    """

    def load_data(self, filename: str) -> pd.DataFrame:
        """
        Load data from an Excel file and filter it based on specific criteria.

        Args:
            filename (str): The path to the Excel file.

        Returns:
            pd.DataFrame: The filtered DataFrame containing questions and answers.

        """
        raw_df = pd.read_excel(filename)

        df_filtered = raw_df.loc[
            raw_df['Kategorie'].str.contains('B') &
            raw_df['Poprawna odp'].isin(['Tak', 'Nie']),
            ['Pytanie', 'Poprawna odp']
        ].reset_index(drop=True)

        df_renamed = df_filtered.rename(columns={'Pytanie': 'question', 'Poprawna odp': 'answer'})
        return df_renamed

    def preprocess_questions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the questions in a DataFrame by lemmatizing, removing stopwords, and punctuation.

        Args:
            df (pd.DataFrame): The DataFrame containing questions and answers.

        Returns:
            pd.DataFrame: The preprocessed DataFrame with original, lemmatized, no_stopwords, no_punctuation, and answer columns.

        """
        tqdm.pandas()

        nlp = NLPModeler()

        questions = df['question']
        answers = df['answer']

        questions_lemmatized = questions.progress_apply(nlp.lemmatize)
        questions_without_stopwords = questions_lemmatized.progress_apply(nlp.remove_stopwords)
        questions_without_punctuation = questions_without_stopwords.progress_apply(nlp.remove_punctuation)

        return pd.DataFrame({
            "original": questions,
            "lemmatized": questions_lemmatized,
            "no_stopwords": questions_without_stopwords,
            "no_punctuation": questions_without_punctuation,
            "answer": answers
        })

    def vectorize(
            self,
            text_col: pd.Series,
            label: pd.Series,
            ngram_range: tuple = (1, 2),
            min_df: int = 5
    ) -> (pd.DataFrame, pd.Series):
        
        """
        Vectorize the text column using CountVectorizer and return the feature matrix and labels.

        Args:
            text_col (pd.Series): The text data to be vectorized.
            label (pd.Series): The corresponding labels.
            ngram_range (tuple): The n-gram range for CountVectorizer (default is (1, 2)).
            min_df (int): The minimum document frequency for CountVectorizer (default is 5).

        Returns:
            pd.DataFrame: The feature matrix.
            pd.Series: The labels.
        """

        vectorizer = CountVectorizer(ngram_range=ngram_range, min_df=min_df)
        X = vectorizer.fit_transform(text_col)
        X_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
        y = label.reset_index(drop=True)
        return X_df, y

