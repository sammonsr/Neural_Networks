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
class PricingModel():
    STRING_COLS = [2, 5, 7, 12, 13, 19, 20, 21, 25]
    BOOL_COLS = [6, 9]
    DROP_COLS = [0, -1, -2]

    uniq_vals_per_col = []

    # YOU ARE ALLOWED TO ADD MORE ARGUMENTS AS NECESSARY
    def __init__(self, calibrate_probabilities=False):
        """
        Feel free to alter this as you wish, adding instance variables as
        necessary.
        """
        self.y_mean = None
        self.calibrate = calibrate_probabilities
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
    def _preprocessor(self, X_raw, currently_training):
        """Data preprocessing function.

        This function prepares the features of the data for training,
        evaluation, and prediction.

        Parameters
        ----------
        X_raw : ndarray
            An array, this is the raw data as downloaded

        Returns
        -------
        X: ndarray
            A clean data set that is used for training and prediction.
        """
        # =============================================================
        # YOUR CODE HERE

        # Perform one hot encoding
        X_raw = self._one_hot_encoding_preproc(X_raw, currently_training)

        # Standardisation
        X_raw = preprocessing.StandardScaler().fit_transform(X_raw)

        # Get relevant columns
        relevant_cols = self._relevant_col_indicies()
        current_col_indicies = list(range(len(X_raw[0])))
        cols_to_delete = filter(lambda c: c not in relevant_cols, current_col_indicies)

        # Remove irrelevant cols
        X_raw = np.delete(X_raw, cols_to_delete, axis=1)

        return X_raw

    def _relevant_col_indicies(self):
        cols_for_base_classifier = "drv_age1,vh_age,vh_cyl,vh_din,pol_bonus,vh_sale_begin,vh_sale_end,vh_value,vh_speed,claim_amount,made_claim".split(
            ',')

        train_data_file = open('part3_training_data.csv')
        cols_in_part_3_data = train_data_file.readline().split()
        train_data_file.close()

        relevant_cols = []
        for col in cols_for_base_classifier:
            col_index = cols_in_part_3_data.index(col)
            relevant_cols.append(col_index)

        return relevant_cols

    def _one_hot_encoding_preproc(self, X_raw, currently_training):
        # One hot encoding method: For a string column, get unique values, then use labelbinarizer
        # to set correct cols to 1 or 0
        # Convert strings to bools
        yes_no_map = np.vectorize(lambda a: 1 if a == "Yes" else 0)
        for bool_col in self.BOOL_COLS:
            X_raw = yes_no_map(X_raw[bool_col])
        if currently_training:
            # Setup one-hot encoding
            for col_index in self.STRING_COLS:
                uniq_values = list(set(X_raw[col_index]))
                self.uniq_vals_per_col[col_index] = uniq_values
        # Use one-hot encoding
        for str_col in self.STRING_COLS:
            new_cols = label_binarize(X_raw[:, str_col], classes=self.uniq_vals_per_col[str_col])
            X_raw = np.concatenate((X_raw, new_cols), axis=1)
        # Drop string cols and drop cols
        np.delete(X_raw, self.STRING_COLS, axis=1)
        np.delete(X_raw, self.DROP_COLS, axis=1)
        return X_raw

    # TODO: Remove claims_raw=None
    def fit(self, X_raw, y_raw, claims_raw):
        """Classifier training function.

        Here you will use the fit function for your classifier.

        Parameters
        ----------
        X_raw : ndarray
            This is the raw data as downloaded
        y_raw : ndarray
            A one dimensional array, this is the binary target variable
        claims_raw: ndarray
            A one dimensional array which records the severity of claims

        Returns
        -------
        self: (optional)
            an instance of the fitted model

        """
        # TODO: find out why claims_raw is a separate column
        claims_raw = X_raw[:, -1]

        nnz = np.where(claims_raw != 0)[0]
        self.y_mean = np.mean(claims_raw[nnz])

        X_clean = self._preprocessor(X_raw, True)

        # THE FOLLOWING GETS CALLED IF YOU WISH TO CALIBRATE YOUR PROBABILITES
        if self.calibrate:
            self.base_classifier = fit_and_calibrate_classifier(
                self.base_classifier, X_clean, y_raw)
        else:
            self.base_classifier = self.base_classifier.fit(X_clean, y_raw)
        return self.base_classifier

    def predict_claim_probability(self, X_raw):
        """Classifier probability prediction function.

        Here you will implement the predict function for your classifier.

        Parameters
        ----------
        X_raw : ndarray
            This is the raw data as downloaded

        Returns
        -------
        ndarray
            A one dimensional array of the same length as the input with
            values corresponding to the probability of beloning to the
            POSITIVE class (that had accidents)
        """
        # Convert pandas dataframe to numpy array
        X_raw = X_raw.numpy()

        # Preprocess data
        X_clean = self._preprocessor(X_raw, False)

        # Need to convert numpy to pandas in order to use predict
        X_as_pandas = pd.DataFrame(X_clean)
        predictions = self.base_classifier.predict(X_as_pandas)

        return predictions

    def predict_premium(self, X_raw):
        """Predicts premiums based on the pricing model.

        Here you will implement the predict function for your classifier.

        Parameters
        ----------
        X_raw : pandas.DataFrame
            A pandas dataframe, this is the raw data as downloaded

        Returns
        -------
        numpy.ndarray
            A one dimensional array of the same length as the input with
            values corresponding to the probability of beloning to the
            POSITIVE class (that had accidents)
        """
        X_raw = X_raw.numpy()

        # =============================================================
        # TODO: REMEMBER TO INCLUDE ANY PRICING STRATEGY HERE.
        # For example you could scale all your prices down by a factor

        return self.predict_claim_probability(X_raw) * self.y_mean

    def save_model(self):
        """Saves the class instance as a pickle file."""
        # =============================================================
        with open('part3_pricing_model_linear.pickle', 'wb') as target:
            pickle.dump(self, target)


def load_data(has_header=True, shuffle=False):
    filename = 'part3_training_data.csv'
    skip_rows = 1 if has_header else 0
    data = np.loadtxt(filename, delimiter=',', skiprows=skip_rows, dtype='O')
    if shuffle:
        np.random.shuffle(data)
    # Split into x and y
    X, y = np.split(data, [-1], axis=1)

    return X, y


def load_model():
    # Please alter this section so that it works in tandem with the save_model method of your class
    with open('part3_pricing_model_linear.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    return trained_model


if __name__ == "__main__":
    model = PricingModel()
    X_train, y_train = load_data()

    model.fit(X_train, y_train)
