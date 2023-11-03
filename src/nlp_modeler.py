import string
import spacy

class NLPModeler:
    """
    A class for performing natural language processing tasks.

    Attributes:
        model (str): The name of the Spacy language model to load (default is 'pl_core_news_lg').

    Methods:
        __init__(self, model: str = 'pl_core_news_lg') -> None:
            Initialize the NLPModeler with a specific Spacy language model.

        remove_stopwords(self, text: str) -> str:
            Remove stopwords from a text.

        remove_punctuation(self, text: str) -> str:
            Remove punctuation from a text.

        lemmatize(self, text: str) -> str:
            Lemmatize a text.

    Note: This class assumes a Spacy language model with stop words is available for the specified language model.
    """

    def __init__(self, model: str = 'pl_core_news_lg') -> None:
        """
        Initialize the NLPModeler with a specific Spacy language model.

        Args:
            model (str): The name of the Spacy language model to load (default is 'pl_core_news_lg').

        Returns:
            None
        """

        self.nlp = spacy.load(model)
        self.stopwords = set(self.nlp.Defaults.stop_words)

    def remove_stopwords(self, text: str) -> str:
        """
        Remove stopwords from a text.

        Args:
            text (str): The input text.

        Returns:
            str: The text with stopwords removed.
        """

        words_list = text.split()
        filtered_words = [word for word in words_list if word not in self.stopwords]
        return ' '.join(filtered_words)

    def remove_punctuation(self, text: str) -> str:
        """
        Remove punctuation from a text.

        Args:
            text (str): The input text.

        Returns:
            str: The text with punctuation removed.
        """

        translator = str.maketrans('', '', string.punctuation)
        text_without_punctuation = text.translate(translator)
        return text_without_punctuation

    def lemmatize(self, text: str) -> str:
        """
        Lemmatize a text.

        Args:
            text (str): The input text.

        Returns:
            str: The lemmatized text.
        """

        doc = self.nlp(text)
        lemmatized_text = [token.lemma_ for token in doc]
        return ' '.join(lemmatized_text)