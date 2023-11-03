from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def generate_wordcloud(
        text: str,
        width: int = 1200,
        height: int = 600,
        background_color: str = 'white'
):
    """
    Generate and display a word cloud from the given text.

    Args:
        text (str): The input text to generate the word cloud from.
        width (int): The width of the word cloud image (default is 1200).
        height (int): The height of the word cloud image (default is 600).
        background_color (str): The background color of the word cloud image (default is 'white').

    Returns:
        None
    """
    wordcloud = WordCloud(
        width=width,
        height=height,
        background_color=background_color
    ).generate(text)

    plt.figure(figsize=(16, 10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

def preprocess_for_fasttext(
        df: pd.DataFrame,
        category: str = 'answer',
        description: str = 'question'
) -> pd.Series:
    """
    Preprocess the data in the given DataFrame for FastText text classification.

    Args:
        df (pd.DataFrame): The input DataFrame containing category and description columns.
        category (str): The column name for the category labels.
        description (str): The column name for the text descriptions.

    Returns:
        pd.Series: A processed series with text data suitable for FastText classification.
    """
    category_description = '__label__' + df[category] + ' ' + df[description]
    category_description = (
        category_description
        .str.replace(r'[^\w\s\']', ' ', regex=True)
        .str.replace(' +', ' ', regex=True)
        .str.strip()
        .str.lower()
    )
    return category_description # pd.series
