# pyright: basic
from sklearn.metrics import confusion_matrix

from lib import utils

p_v = "./sentiment_analysis_validation_data.csv"
p_test = "./sentiment_analysis_test_data.csv"
p_t = "./sentiment_analysis_training_data.csv"


def get_confusion_matrix(true_label, pred_label):
    """
    Calculate the confusion matrix for your predicted labels. See https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    :param pred_label: Array of predicted labels
    :param true_label: Array of corresponding ground truth (test) labels
    :return: Confusion matrix whose i-th row and j-th column entry indicates the number of samples with true label being i-th class and predicted label being j-th class.
    """
    return confusion_matrix(true_label, pred_label)


def run_task1():
    text_train, labels_train, text_val, labels_val, text_test = utils.load_data_train(
        path_val=p_v, path_test=p_test, path_train=p_t
    )
