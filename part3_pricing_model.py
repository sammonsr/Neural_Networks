import math

import sklearn
import sys

from sklearn import preprocessing
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
from sklearn.preprocessing import label_binarize
import pandas as pd
import part2_claim_classifier
# Import needed for load_model in part 2 to work
from part2_claim_classifier import ClaimClassifier


def fit_and_calibrate_classifier(classifier, X, y):
    # DO NOT ALTER THIS FUNCTION
    X_train, X_cal, y_train, y_cal = train_test_split(
        X, y, train_size=0.85, random_state=0)
    classifier = classifier.fit(X_train, y_train)

    # This line does the calibration for you
    calibrated_classifier = CalibratedClassifierCV(
        classifier, method='sigmoid', cv='prefit').fit(X_cal, y_cal)
    return calibrated_classifier


# class for part 3
class PricingModel:
    STRING_COLS = [2, 5, 7, 12, 13, 19, 20, 25, 33]
    BOOL_COLS = [6, 9]
    DROP_COLS = [0, 8, 21, 28, 29, 30, 31, 32, 34]

    # YOU ARE ALLOWED TO ADD MORE ARGUMENTS AS NECESSARY
    def __init__(self, calibrate_probabilities=False):
        """
        Feel free to alter this as you wish, adding instance variables as
        necessary.
        """
        self.y_mean = None
        self.calibrate = calibrate_probabilities
        self.uniq_vals_per_col = {}
        # =============================================================
        # READ ONLY IF WANTING TO CALIBRATE
        # Place your base classifier here
        # NOTE: The base estimator must have:
        #    1. A .fit method that takes two arguments, X, y
        #    2. Either a .predict_proba method or a decision
        #       function method that returns classification scores
        #
        # Note that almost every classifier you can find has both.
        # If the one you wish to use does not then speak to one of the TAs
        #
        # If you wish to use the classifier in part 2, you will need
        # to implement a predict_proba for it before use
        # =============================================================
        self.base_classifier = part2_claim_classifier.load_model()

    # YOU ARE ALLOWED TO ADD MORE ARGUMENTS AS NECESSARY TO THE _preprocessor METHOD
    def _preprocessor(self, X_raw, y_raw, currently_training):
        """Data preprocessing function.

        This function prepares the features of the data for training,
        evaluation, and prediction.

        Parameters
        ----------
        X_raw : pandas.DataFramebase_classifier
            An array, this is the raw data as downloaded

        Returns
        -------
        X: ndarray
            A clean data set that is used for training and prediction.
        """
        # =============================================================
        # YOUR CODE HERE
        print("Preprocessing", "training data" if currently_training else "prediction data")

        # Convert pandas dataframe to numpy array
        X_raw = X_raw.to_numpy()

        # Perform one hot encoding
        X_raw = self._one_hot_encoding_preproc(X_raw, currently_training)

        if currently_training:
            y_raw = y_raw.to_numpy()
            # Removes rows with missing data to prevent filling with fake or biased data. Aprrox. 400 rows
            X_raw, y_raw = self._remove_data_if_missing_values(X_raw, y_raw)
            print("Shape of data: ", X_raw.shape)

        # Standardisation
        X_raw = preprocessing.StandardScaler().fit_transform(X_raw)

        return X_raw, y_raw

    def perform_hyper_param_tuning(self, X, y):
        # Preprocess data
        X, y = self._preprocessor(X, y, True)

        train_X_raw, train_y_raw, test_X_raw, test_y_raw, validation_X_raw, validation_y_raw = part2_claim_classifier.get_data_split(
            X, y)

        return part2_claim_classifier.ClaimClassifierHyperParameterSearch(train_X_raw, train_y_raw, validation_X_raw,
                                                                          validation_y_raw,
                                                                          preprocess=False)

    def _remove_data_if_missing_values(self, X_raw, y_raw):
        bad_rows = []
        for i, row in enumerate(X_raw):
            for col in range(len(row)):
                if row[col] == '' or row[col] is None or math.isnan(row[col]):
                    bad_rows.append(i)
                    break
        return np.delete(X_raw, bad_rows, axis=0), np.delete(y_raw, bad_rows, axis=0)

    def _one_hot_encoding_preproc(self, X_raw, currently_training):
        # One hot encoding method: For a string column, get unique values, then use labelbinarizer
        # to set correct cols to 1 or 0
        # Convert strings to bools
        yes_no_map = np.vectorize(lambda a: 1 if a == "Yes" else 0)
        for bool_col in self.BOOL_COLS:
            X_raw[:, bool_col] = yes_no_map(X_raw[:, bool_col])

        fill_empty_map = np.vectorize(lambda a: "___BLANK___" if a is None else a, otypes=[str])
        for col_index in self.STRING_COLS:
            X_raw[:, col_index] = fill_empty_map(X_raw[:, col_index])

        if currently_training:
            # Setup one-hot encoding
            for col_index in self.STRING_COLS:
                uniq_values = list(set(X_raw[:, col_index]))
                self.uniq_vals_per_col[col_index] = uniq_values

        # Use one-hot encoding
        for str_col in self.STRING_COLS:
            new_cols = label_binarize(X_raw[:, str_col], classes=self.uniq_vals_per_col[str_col])
            X_raw = np.concatenate((X_raw, new_cols), axis=1)

        # Drop string cols and drop cols
        X_raw = np.delete(X_raw, self.STRING_COLS + self.DROP_COLS, axis=1)

        # If any row contains an empty cell, delete row
        return X_raw

    def fit(self, X_raw, claims_raw, y_raw):
        """Classifier training function.

        Here you will use the fit function for your classifier.

        Parameters
        ----------
        X_raw : pandas.DataFrame
            This is the raw data as downloaded
        y_raw : pandas.DataFrame
            A one dimensional array, this is the binary target variable
        claims_raw: pandas.DataFrame
            A one dimensional array which records the severity of claims

        Returns
        -------
        self: (optional)
            an instance of the fitted model

        """

        nnz = np.where(claims_raw.to_numpy() != 0)[0]
        self.y_mean = np.mean(claims_raw.to_numpy()[nnz])

        X_clean, y_clean = self._preprocessor(X_raw, y_raw, True)

        print("Fitting model")

        # THE FOLLOWING GETS CALLED IF YOU WISH TO CALIBRATE YOUR PROBABILITIES
        if self.calibrate:
            self.base_classifier = fit_and_calibrate_classifier(
                self.base_classifier, X_clean, y_clean)
        else:
            self.base_classifier = self.base_classifier.fit(X_clean, y_clean, preprocess=False)
        return self.base_classifier

    def predict_claim_probability(self, X_raw):
        """Classifier probability prediction function.

        Here you will implement the predict function for your classifier.

        Parameters
        ----------
        X_raw : pandas.DataFrame
            This is the raw data as downloaded

        Returns
        -------
        ndarray
            A one dimensional array of the same length as the input with
            values corresponding to the probability of beloning to the
            POSITIVE class (that had accidents)
        """
        # Preprocess data
        X_clean, _ = self._preprocessor(X_raw, None, False)

        predictions = self.base_classifier.predict(X_clean, preprocess=False)

        return predictions

    def predict_premium(self, X_pandas):
        """Predicts premiums based on the pricing model.

        Here you will implement the predict function for your classifier.

        Parameters
        ----------
        X_pandas : pandas.DataFrame
            A pandas dataframe, this is the raw data as downloaded

        Returns
        -------
        numpy.ndarray
            A one dimensional array of the same length as the input with
            values corresponding to the probability of beloning to the
            POSITIVE class (that had accidents)
        """

        # =============================================================
        # TODO: REMEMBER TO INCLUDE ANY PRICING STRATEGY HERE.
        # For example you could scale all your prices down by a factor

        probabilities = self.predict_claim_probability(X_pandas)[:, 0]

        MIN = 0
        MAX = 0.25

        print("Max=", max(probabilities))
        print("Min=", min(probabilities))

        scaled_probabilities = ((probabilities - MIN) / (MAX - MIN))

        return scaled_probabilities * self.y_mean

    def save_model(self):
        """Saves the class instance as a pickle file."""
        # =============================================================
        with open('part3_pricing_model_linear.pickle', 'wb') as target:
            pickle.dump(self, target)


def load_data(has_header=True, shuffle=False):
    filename = 'part3_training_data.csv'
    skip_rows = 1 if has_header else 0
    data = pd.read_csv(filename)
    if shuffle:
        data = sklearn.utils.shuffle(data)

    # Split into x and y
    X = data.iloc[:, :-2]
    claims = data.iloc[:, -2:-1]
    y = data.iloc[:, -1:]

    return X, claims, y


def load_model():
    # Please alter this section so that it works in tandem with the save_model method of your class
    with open('part3_pricing_model_linear.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    return trained_model


if __name__ == "__main__":
    model = PricingModel(False)
    X_train, claim_train, y_train = load_data()

    #print(model.perform_hyper_param_tuning(X_train, y_train))

    # Convert data into dataframe
    X_train = pd.DataFrame(X_train)
    claim_train = pd.DataFrame(claim_train)
    y_train = pd.DataFrame(y_train)

    # Train model
    model.fit(X_train, claim_train, y_train)

    # Test by performing predictions
    bad_rows = []
    for i, row in enumerate(X_train):
        for col in range(len(row)):
            if row[col] == '' or row[col] is None:
                bad_rows.append(i)
                break
    X_pred = np.delete(X_train, bad_rows, axis=0)

    predictions = model.predict_premium(pd.DataFrame(X_pred))

    # Save model
    model.save_model()

    print(predictions)
