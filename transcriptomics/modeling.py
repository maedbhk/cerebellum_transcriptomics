import numpy as np

from sklearn.model_selection import train_test_split
import sklearn.linear_model as lm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import confusion_matrix
from sklearn import tree
from sklearn import cluster
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.preprocessing import label_binarize

from transcriptomics.constants import Defaults
from transcriptomics.visualization import visualize
from transcriptomics.gec_functions_preprocess import save_thresholded_data

def _split_train_test(X, y, test_size):
    """ divides X and y into train and test sets
        Args:
            X (matrix): training data
            y (vector): labelled data
            test_size (int): size of test size, usually .2
        Returns:
            x_train, x_test, y_train, y_test
    """
    # y = label_binarize(y, classes)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    return X_train, X_test, y_train, y_test

def _run_classifier_multiclass(X, y, classifier='logistic', test_size=.2): 
    """ fit logistic regression model - crossvalidated
        Args:
            X_train (matrix):
            y_train (vector):
            classifier (str): 'logistic', 'svm' etc
            test_size (int): default is .2
            fit_intercept (bool): default is True

            Kwargs:
                solver (str): default is 'lbfgs'
        Return:
            logistic model
    """
    random_state = np.random.seed(47)

    X_train, X_test, y_train, y_test = _split_train_test(X, y, test_size=test_size)

    if classifier=='logistic':
        classifier = lm.LogisticRegression(fit_intercept=True, multi_class='ovr', solver='lbfgs')  #multi_class='ovr'  
        classifier.fit(X_train, y_train)
        classifier_probs = classifier.predict_proba(X_test)
    elif classifier=="svm":
        classifier = OneVsRestClassifier(svm.LinearSVC(random_state=random_state))
        classifier.fit(X_train, y_train)
        classifier_probs = classifier.decision_function(X_test)

    # predict class values
    y_pred = classifier.predict(X_test)

    return classifier, classifier_probs, y_pred, y_test

def _run_classifier_binary(X, y, classifier='logistic', test_size=.2):
    """ fit logistic regression model - crossvalidated
        Args:
            X_train (matrix):
            y_train (vector):
            classifier (str): 'logistic', 'svm' etc
            test_size (int): default is .2
            fit_intercept (bool): default is True

            Kwargs:
                solver (str): default is 'lbfgs'
        Return:
            logistic model
    """
    random_state = np.random.seed(47)

    X_train, X_test, y_train, y_test = _split_train_test(X, y, test_size=test_size)

    if classifier=='logistic':
        classifier = lm.LogisticRegression(fit_intercept=True, solver='lbfgs') # solver='lbfgs'
        classifier.fit(X_train, y_train)
        classifier_probs = classifier.predict_proba(X_test)
        # keep probabilities for the positive outcome only
        classifier_probs = classifier_probs[:,1]
    elif classifier=="svm":
        classifier = svm.LinearSVC(random_state=random_state)
        classifier.fit(X_train, y_train)
        classifier_probs = classifier.decision_function(X_test)

    # predict class values
    y_pred = classifier.predict(X_test)

    return classifier, classifier_probs, y_pred, y_test

def _confusion_matrix(X, y, classifier='logistic', label_type='multi-class', test_size=.2):
    """ Returns confusion matrix for either accuracy or precision_recall
        Args:
            X (matrix): training data
            y (vector): labelled data
            test_size (int): default is .w
            type (str): "accuracy" or "precision_recall"
        Returns:
            confusion matrix
    """
    # run classifier
    if label_type=="multi-class":
        _, _, y_pred, y_test = _run_classifier_multiclass(X, y, classifier, test_size)
        f1 = f1_score(y_test, y_pred, average='micro')
    elif label_type=="binary":
        _, _, y_pred, y_test = _run_classifier_binary(X, y, classifier, test_size)
        f1 = f1_score(y_test, y_pred)
    else: 
        print(f'{label_type} does not exist. options are multi-class or binary')

    cnf_matrix = confusion_matrix(y_test, y_pred)

    return cnf_matrix, f1

def _recall_precision(X, y, classifier='logistic', label_type='multi-class', test_size=.2):
    if label_type=="multi-class":
        # run classifier on multi-class data
        classifier, classifier_probs, y_pred, y_test = _run_classifier_multiclass(X, y, classifier, test_size)

        # calculate precision and recall for multi-class
        y_binarized = label_binarize(y_test, y_test.unique())
        n_classes = y_binarized.shape[1]
        precision = dict()
        recall = dict()
        average_precision = dict()
        for i in range(n_classes):
            precision[i], recall[i], _ = precision_recall_curve(y_binarized[:, i], classifier_probs[:, i])
            average_precision[i] = average_precision_score(y_binarized[:, i], classifier_probs[:, i])

        # A "micro-average": quantifying score on all classes jointly
        precision["micro"], recall["micro"], _ = precision_recall_curve(y_binarized.ravel(),
        classifier_probs.ravel())
        average_precision["micro"] = average_precision_score(y_binarized, classifier_probs,
                                                        average="micro")
        f1 = f1_score(y_test, y_pred, average='micro')
        # f1 = f1_score(y_test, y_pred, average=None) # f1 for all classes

    elif label_type=="binary":
        # run classifier on binary data
        classifier, classifier_probs, y_pred, y_test = _run_classifier_binary(X, y, classifier, test_size)

        precision, recall, _ = precision_recall_curve(y_test, classifier_probs) 
        average_precision = average_precision_score(y_test, classifier_probs) 
        n_classes = 1 
        f1, auc_score = f1_score(y_test, y_pred), auc(recall, precision)            
    else:
        print(f'{label_type} does not exist. options are multi-class or binary')               

    return precision, recall, average_precision, n_classes, f1

def _get_X_y(atlas, dataframe, label_type):
    """ returns training data (X), labelled data (Y), and classes
        Args:
            dataframe: dataframe containing X, Y, and classes
            atlas (str): which atlas are we working with? defines classes based on atlas
            label_type (str): 'multi-class' or 'binary'
        Returns:
            X, y, classes
    """
    np.random.seed(10)
    # x_cols = _get_gene_symbols(atlas="MDTB-10", which_genes='top', percentile=1)
    x_cols = dataframe.filter(regex=("[A-Z0-9].*")).columns

    if label_type=="binary":
        if atlas=="MDTB-10-subRegions":
            dataframe['class_num'] = dataframe['region_num'].apply(lambda x: 0 if x<11 else 1)
            dataframe['class_name'] = dataframe['region_id'].str.extract(r'-(\w+)')
        else:
            print(f'binary option not available for {atlas}')

    if label_type=="multi-class":
            dataframe['class_num'] = dataframe['region_num']
            dataframe['class_name'] = dataframe['region_id']

    X = dataframe[x_cols]
    y = dataframe['class_num']

    classes = dataframe['class_name'].unique()

    return X, y, classes

def _compute_CV_error(model, X, y, test_size=.2):
    '''
    Split the training data into 4 subsets.
    For each subset, 
        fit a model holding out that subset
        compute the MSE on that subset (the validation set)
    You should be fitting 4 models total.
    Return the average MSE of these 4 folds.

    Args:
        model: an sklearn model with fit and predict functions 
        X_train (data_frame): Training data
        y_train (data_frame): Label 

    Return:
        the average validation MSE for the 4 splits.
    '''
    np.random.seed(10)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    kf = KFold(n_splits=4)
    validation_errors = []
    
    for train_idx, valid_idx in kf.split(X_train):

        # split the data
        split_X_train, split_X_valid = X_train.iloc[train_idx], X_train.iloc[valid_idx]
        split_y_train, split_y_valid = y_train.iloc[train_idx], y_train.iloc[valid_idx]
      
        # Fit the model on the training split
        model.fit(split_X_train, split_y_train)
        
        error = _rmse(model.predict(split_X_valid), split_y_valid)

        validation_errors.append(error)
        
    return np.mean(validation_errors)

def _compute_test_train_error(model, X, y, test_size=.2):
    np.random.seed(10)

    train_error_vs_N = []
    test_error_vs_N = []

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    range_of_num_features = range(1, X_train.shape[1] + 1)

    for N in range_of_num_features:
        X_train_first_N_features = X_train.iloc[:, :N]    
        
        model.fit(X_train_first_N_features, y_train)
        train_error = _rmse(model.predict(X_train_first_N_features), y_train)
        train_error_vs_N.append(train_error)
        
        X_test_first_N_features = X_test.iloc[:, :N]
        test_error = _rmse(model.predict(X_test_first_N_features), y_test)    
        test_error_vs_N.append(test_error)
        print(f'test train error computed for {N} features')

    return test_error_vs_N, train_error_vs_N, range_of_num_features

def _rmse(y_pred, y_test, test_size=.2):
    """
    Args:
        y_pred: an array of the prediction from the model
        y_test: an array of the groudtruth label
        
    Returns:
        The root mean square error between the prediction and the groudtruth
    """
    return np.sqrt(np.mean((y_test - y_pred) ** 2)) 

def _get_model(model_type='linear'):
    if model_type=="linear":
        model = lm.LinearRegression()

    return model

def _fit_linear_model_optimal_features(X, y, test_size=.2):
    np.random.seed(10)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    range_of_num_features = range(1, X_train.shape[1] + 1)

    errors = []
    for N in range_of_num_features:
        print(f"Trying first {N} features")
        model = lm.LinearRegression()
        
        # compute the cross validation error
        error = _compute_CV_error(model, X_train.iloc[:, :N], y_train) 
        
        print("\tRMSE:", error)
        errors.append(error)

    best_num_features = np.argmin(errors) + 1
    best_err = errors[best_num_features - 1]

    print(f"Best choice, use the first {best_num_features} features")

    # Fit linear model with best features
    model = lm.LinearRegression()
    model.fit(X_train.iloc[:, :best_num_features], y_train)
    train_rmse = _rmse(model.predict(X_train.iloc[:, :best_num_features]), y_train) 
    test_rmse = _rmse(model.predict(X_test.iloc[:, :best_num_features]), y_test)

    print("Train RMSE", train_rmse)
    print("KFold Validation RMSE", best_err)
    print("Test RMSE", test_rmse)

    return best_num_features, train_rmse, best_err, test_rmse