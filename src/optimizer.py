from typing import Any, Dict
import optuna
from sklearn.tree import DecisionTreeClassifier
from optuna.samplers import RandomSampler
import pandas as pd
from src.modeler import Modeler

class Optimizer(Modeler):
    """
    A class for optimizing machine learning model hyperparameters using Optuna.

    Attributes:
        algorithm (Any): The machine learning algorithm to optimize.
        input_parameters (Dict[str, Tuple[str, Union[int, float, List[Union[int, float, str]]]]):
            A dictionary of hyperparameters to optimize.
        X (pd.DataFrame): The feature matrix.
        y (pd.Series): The target labels.
        split (bool): Whether to split the data into training and testing sets (default is False).
        test_size (float): The size of the test set when splitting the data (default is 0).

    Methods:
        __init__(self, algorithm: Any, input_parameters: Dict[str, Tuple[str, Union[int, float, List[Union[int, float, str]]]],
                 X: pd.DataFrame, y: pd.Series, split: bool = False, test_size: float = 0) -> None:
            Initialize the Optimizer with an algorithm, hyperparameters, and data.

        _convert_parameters(self, input_parameters: Dict[str, Tuple[str, Union[int, float, List[Union[int, float, str]]]],
                          trial: optuna.trial.Trial) -> Dict[str, Any]:
            Convert hyperparameter suggestions to a dictionary of parameters for model training.

        objective(self, trial: optuna.trial.Trial) -> float:
            Define the objective function for Optuna optimization.

        optimize(self, n_trials: int, verbosity: int = 0) -> Dict[str, Any]:
            Optimize hyperparameters using Optuna and return the best parameters.
    """

    def __init__(
            self,
            algorithm: Any,
            input_parameters: dict,
            X: pd.DataFrame,
            y: pd.Series,
            split: bool = False,
            test_size: float = 0
    ) -> None:
        """
        Initialize the Optimizer with an algorithm, hyperparameters, and data.

        Args:
            algorithm (Any): The machine learning algorithm to optimize.
            input_parameters (Dict):
                A dictionary of hyperparameters to optimize.
            X (pd.DataFrame): The feature matrix.
            y (pd.Series): The target labels.
            split (bool): Whether to split the data into training and testing sets (default is False).
            test_size (float): The size of the test set when splitting the data (default is 0).

        Returns:
            None
        """
        super().__init__(algorithm)
        self.split = split
        self.test_size = test_size
        self.input_parameters = input_parameters
        self.X = X
        self.y = y

    def _convert_parameters(
            self,
            input_parameters: dict,
            trial: optuna.trial.Trial
    ) -> Dict[str, Any]:
        """
        Convert hyperparameter suggestions to a dictionary of parameters for model training.

        Args:
            input_parameters (Dict):
                A dictionary of hyperparameters and their types and value ranges.
            trial (optuna.trial.Trial): Optuna trial object for parameter suggestion.

        Returns:
            Dict[str, Any]: A dictionary of converted hyperparameters.

        Example:
            converted_params = optimizer._convert_parameters(hyperparameters, trial)
        """
        suggest_functions = {
            'int': optuna.trial.Trial.suggest_int,
            'float': optuna.trial.Trial.suggest_float,
            'categorical': optuna.trial.Trial.suggest_categorical
        }
        converted_parameters = {}
        for key, (param_type, param_info) in input_parameters.items():
            suggest_function = suggest_functions.get(param_type)
            if suggest_function:
                param_name, *param_values = param_info
                converted_parameters[key] = suggest_function(trial, param_name, *param_values)
        return converted_parameters
    
    def objective(self, trial: optuna.trial.Trial) -> float:
        """
        Define the objective function for Optuna optimization.

        Args:
            trial (optuna.trial.Trial): Optuna trial object for parameter suggestion.

        Returns:
            float: The objective function value to maximize.

        Example:
            accuracy = optimizer.objective(trial)
        """
    
        parameters = self._convert_parameters(self.input_parameters, trial)
        results = self.fit_predict(
            X=self.X, # TODO cos tu nie dzials jak powinno
            y=self.y,
            split=self.split,
            test_size=self.test_size,
            model_params=parameters
        )
        

        evaluation = self.evaluate(results['y_test'], results['y_pred'])
        accuracy = evaluation['accuracy']
        return accuracy
    
    def optimize(self, n_trials: int, verbosity: int = 0, seed: int = 42) -> Dict[str, Any]:
        """
        Optimize hyperparameters using Optuna and return the best parameters.

        Args:
            n_trials (int): The number of trials to run during optimization.
            verbosity (int): Optuna verbosity level (default is 0).
            seed (int): Random seed (default is 42).

        Returns:
            Dict[str, Any]: The best hyperparameters found by Optuna.
        """
        optuna.logging.set_verbosity(verbosity)
        sampler = RandomSampler(seed=seed)  # Make the sampler behave in a deterministic way.
        study = optuna.create_study(sampler=sampler)
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=n_trials)

        return study.best_params


