from aim5005.features import MinMaxScaler, StandardScaler
import numpy as np
import unittest
from unittest.case import TestCase

### TO NOT MODIFY EXISTING TESTS

class TestFeatures(TestCase):
    def test_initialize_min_max_scaler(self):
        scaler = MinMaxScaler()
        assert isinstance(scaler, MinMaxScaler), "scaler is not a MinMaxScaler object"
        
        
    def test_min_max_fit(self):
        scaler = MinMaxScaler()
        data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
        scaler.fit(data)
        assert (scaler.maximum == np.array([1., 18.])).all(), "scaler fit does not return maximum values [1., 18.] "
        assert (scaler.minimum == np.array([-1., 2.])).all(), "scaler fit does not return maximum values [-1., 2.] " 
        
        
    def test_min_max_scaler(self):
        data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
        expected = np.array([[0., 0.], [0.25, 0.25], [0.5, 0.5], [1., 1.]])
        scaler = MinMaxScaler()
        scaler.fit(data)
        result = scaler.transform(data)
        assert (result == expected).all(), "Scaler transform does not return expected values. All Values should be between 0 and 1. Got: {}".format(result.reshape(1,-1))
        
    def test_min_max_scaler_single_value(self):
        data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
        expected = np.array([[1.5, 0.]])
        scaler = MinMaxScaler()
        scaler.fit(data)
        result = scaler.transform([[2., 2.]]) 
        assert (result == expected).all(), "Scaler transform does not return expected values. Expect [[1.5 0. ]]. Got: {}".format(result)
        
    def test_standard_scaler_init(self):
        scaler = StandardScaler()
        assert isinstance(scaler, StandardScaler), "scaler is not a StandardScaler object"
        
    def test_standard_scaler_get_mean(self):
        scaler = StandardScaler()
        data = [[0, 0], [0, 0], [1, 1], [1, 1]]
        expected = np.array([0.5, 0.5])
        scaler.fit(data)
        assert (scaler.mean == expected).all(), "scaler fit does not return expected mean {}. Got {}".format(expected, scaler.mean)
        
    def test_standard_scaler_transform(self):
        scaler = StandardScaler()
        data = [[0, 0], [0, 0], [1, 1], [1, 1]]
        expected = np.array([[-1., -1.], [-1., -1.], [1., 1.], [1., 1.]])
        scaler.fit(data)
        result = scaler.transform(data)  # Assign the transformed data to result
        assert (result == expected).all(), "Scaler transform does not return expected values. Expect {}. Got: {}".format(expected.reshape(1, -1), result.reshape(1, -1))

    
    def test_standard_scaler_single_value(self):
        data = [[0, 0], [0, 0], [1, 1], [1, 1]]
        expected = np.array([[3., 3.]])
        scaler = StandardScaler()
        scaler.fit(data)
        result = scaler.transform([[2., 2.]]) 
        assert (result == expected).all(), "Scaler transform does not return expected values. Expect {}. Got: {}".format(expected.reshape(1,-1), result.reshape(1,-1))

    # TODO: Add a test of your own below this line

    def test_standard_scaler_zero_variance_mixed(self):
        scaler = StandardScaler()
        data = np.array([[3, 5], [3, 10], [3, 15], [3, 20]])  # First column has zero variance
        scaler.fit(data)

        # Corrected expected values based on actual standard deviation calculations
        expected = np.array([[0, -1.34164079], [0, -0.4472136], [0, 0.4472136], [0, 1.34164079]])  
        result = scaler.transform(data)

        # Check that the first column is all zeros (due to zero variance)
        assert (result[:, 0] == 0).all(), "Scaler did not handle zero variance feature correctly. Expected all zeros in the first column."

        # Check that the second column is correctly standardized
        assert np.allclose(result[:, 1], expected[:, 1]), "Scaler did not standardize the second column as expected. Got: {}".format(result[:, 1])


    class TestLabelEncoder:
        def test_fit_alternative(self):
            le = LabelEncoder()
            # Using different city names for fitting
            le.fit(["berlin", "berlin", "madrid", "london", "london"])
            assert np.array_equal(le.classes_, ["berlin", "london", "madrid"]), "Classes do not match expected values."

        def test_transform_alternative(self):
            le = LabelEncoder()
            # Fitting on different labels
            le.fit(["apple", "banana", "cherry"])
            transformed = le.transform(["banana", "cherry", "apple", "banana"])
            assert np.array_equal(transformed, [1, 2, 0, 1]), "Transform output does not match expected values."

        def test_fit_transform_alternative(self):
            le = LabelEncoder()
            # Fitting and transforming on different labels
            transformed = le.fit_transform(["dog", "cat", "bird", "dog", "bird"])
            assert np.array_equal(transformed, [2, 1, 0, 2, 0]), "Fit transform output does not match expected values."

        def test_transform_unseen_alternative(self):
            le = LabelEncoder()
            le.fit(["red", "green", "blue"])
            # Check transforming an unseen label raises an error
            with pytest.raises(ValueError):
                le.transform(["purple"])

    
if __name__ == '__main__':
    unittest.main()