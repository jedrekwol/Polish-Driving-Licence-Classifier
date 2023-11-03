from typing import Dict, Any
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import pandas as pd


class Modeler:
    """
    A class for fitting, predicting, and evaluating machine learning models.

    Attributes:
        algorithm (Any): The machine learning algorithm to use for modeling.

    Methods:
        __init__(self, algorithm: Any = None) -> None:
            Initialize the Modeler with a specific machine learning algorithm.

        fit_predict(self, X: pd.DataFrame, y: pd.Series, model_params: Dict[str, Any],
                    split: bool = False, test_size: float = 0, random_state: int = 42) -> Dict[str, Any]:
            Fit a machine learning model, optionally split the data, and make predictions.

        evaluate(self, y_test: pd.Series, y_pred: pd.Series, pos_label: str = 'Tak') -> Dict[str, float]:
            Evaluate the model's performance using various metrics.
    """

    def __init__(self, algorithm: Any = None) -> None:
        """
        Initialize the Modeler with a specific machine learning algorithm.

        Args:
            algorithm (Any): The machine learning algorithm to use for modeling.

        Returns:
            None
        """
        self.algorithm = algorithm


    def fit_predict(
            self,
            X: pd.DataFrame,
            y: pd.Series,
            model_params: Dict[str, Any],
            split: bool = False,
            test_size: float = 0,
            random_state: int = 42
    ) -> Dict[str, Any]:
        """
        Fit a machine learning model, optionally split the data, and make predictions.

        Args:
            X (pd.DataFrame): The feature matrix.
            y (pd.Series): The target labels.
            model_params (Dict[str, Any]): model_params for the machine learning algorithm.
            split (bool): Whether to split the data into training and testing sets (default is False).
            test_size (float): The size of the test set when splitting the data (default is 0).
            random_state (int): Random seed for reproducibility (default is 42).

        Returns:
            Dict[str, Any]: A dictionary containing training and testing data, as well as model predictions.
        """
        self.classifier = self.algorithm(random_state=random_state, **model_params)
        if split:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        else:
            X_train, y_train, X_test, y_test = X, y, X, y
        self.classifier.fit(X_train, y_train)
        y_pred = self.classifier.predict(X_test)   # DO PRZEKMNIENIA JAK PRZEKAZYWAC X, CZY JUZ PO TRAIN TEST SPLIT CZY NIE - CHYBA PO PROSTU DAWAC OSOBNO
        
        results = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'y_pred': y_pred
        }
        return results
        

    def evaluate(
            self,
            y_test: pd.Series,
            y_pred: pd.Series,
            pos_label: str = 'Tak'
    ) -> Dict[str, float]:
        """
        Evaluate the model's performance using various metrics.

        Args:
            y_test (pd.Series): The true target labels.
            y_pred (pd.Series): The predicted target labels.
            pos_label (str): The positive label for precision and recall calculation (default is 'Tak').

        Returns:
            Dict[str, float]: A dictionary containing evaluation metrics.
        """

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, pos_label=pos_label)
        recall = recall_score(y_test, y_pred, pos_label=pos_label)
        f1 = f1_score(y_test, y_pred, pos_label=pos_label)
        
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
        
        metrics_rounded = {key: round(value, 2) for key, value in metrics.items()}
        return metrics_rounded
