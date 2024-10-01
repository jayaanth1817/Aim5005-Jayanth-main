import numpy as np
from typing import List, Tuple
### YOU MANY NOT ADD ANY MORE IMPORTS (you may add more typing imports)

class MinMaxScaler:
    def __init__(self):
        self.minimum = None
        self.maximum = None
        
    def _check_is_array(self, x:np.ndarray) -> np.ndarray:
        """
        Try to convert x to a np.ndarray if it'a not a np.ndarray and return. If it can't be cast raise an error
        """
        if not isinstance(x, np.ndarray):
            x = np.array(x)
            
        assert isinstance(x, np.ndarray), "Expected the input to be a list"
        return x
        
    
    def fit(self, x:np.ndarray) -> None:   
        x = self._check_is_array(x)
        self.minimum=x.min(axis=0)
        self.maximum=x.max(axis=0)
        
    def transform(self, x:np.ndarray) -> list:
        """
        MinMax Scale the given vector
        """
        x = self._check_is_array(x)
        diff_max_min = self.maximum - self.minimum
        
        # TODO: There is a bug here... Look carefully! 
        return (x - self.minimum) / (self.maximum - self.minimum)
    
    def fit_transform(self, x:list) -> np.ndarray:
        x = self._check_is_array(x)
        self.fit(x)
        return self.transform(x)
    
    
class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        """
        Compute the mean and standard deviation of each feature.
        """
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        return self

    def transform(self, X):
        if self.mean is None or self.std is None:
            raise RuntimeError("This StandardScaler instance is not fitted yet.")
        std_adj = np.where(self.std == 0, 1, self.std)  # Prevent division by zero
        return (X - self.mean) / std_adj

    def fit_transform(self, X):
        """
        Fit to data, then transform it.
        """
        self.fit(X)
        return self.transform(X)

    

class LabelEncoder:
    def __init__(self):
        self.classes_ = None

    def fit(self, y):
        """
        Determine the unique classes in the input labels.
        """
        self.classes_ = np.unique(y)
        return self

    def transform(self, y):
        """
        Convert the input labels to their corresponding integer values.
        """
        if self.classes_ is None:
            raise ValueError("This LabelEncoder instance hasn't been fitted yet.")
        return np.array([np.where(self.classes_ == label)[0][0] for label in y])

    def fit_transform(self, y):
        """
        Fit the encoder to the labels and then transform them.
        """
        return self.fit(y).transform(y)

