# pyright: basic
import regex as re  # https://en.wikipedia.org/wiki/Perl_Compatible_Regular_Expressions
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
    detect_spam(text_train)

    # Detect spam
    # Classify remaining reviews as pos / neg
    # Handle noisy labels since spam was in both


# https://johndalesandro.com/blog/comprehensive-regex-for-url-detection-and-spam-filtering/
# stolen_regex = r"~(?(DEFINE)(?<HOSTNAME>(?:(?:[\p{L}\p{N}\p{M}](?![\p{L}\p{N}\p{M}-]{1,2}--)(?:[\p{L}\p{N}\p{M}-]{0,61}[\p{L}\p{N}\p{M}])?|xn--(?![A-Za-z0-9]{2}--)(?:[A-Za-z0-9-]{1,59}[A-Za-z0-9])?|[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)(?:\.(?:[\p{L}\p{N}\p{M}](?![\p{L}\p{N}\p{M}-]{1,2}--)(?:[\p{L}\p{N}\p{M}-]{0,61}[\p{L}\p{N}\p{M}])?|xn--(?![A-Za-z0-9]{2}--)(?:[A-Za-z0-9-]{1,59}[A-Za-z0-9])?|[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?))+\.?)))(?<![\p{L}\p{N}\p{M}_])(?:(?:(?:https?|ftps?)://(?:[\p{L}\p{N}\p{M}\-._\~!$&'()*+,;=%]+(?::[\p{L}\p{N}\p{M}\-._\~!$&'()*+,;=%]*)?@)?(?:localhost|(?&HOSTNAME)|(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(?:\.(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)){3})|\[[0-9A-Fa-f:]+\]))|(?&HOSTNAME))(?::\d{1,5})?(?:[/?#](?:(?:%[0-9A-Fa-f]{2})|[A-Za-z0-9\-._\~]|[\p{L}\p{N}\p{M}\p{S}\p{P}])*)?(?![\p{L}\p{N}\p{M}_])~isug"
# No delims here chief
stolen_regex = r"(?(DEFINE)(?<HOSTNAME>(?:(?:[\p{L}\p{N}\p{M}](?![\p{L}\p{N}\p{M}-]{1,2}--)(?:[\p{L}\p{N}\p{M}-]{0,61}[\p{L}\p{N}\p{M}])?|xn--(?![A-Za-z0-9]{2}--)(?:[A-Za-z0-9-]{1,59}[A-Za-z0-9])?|[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)(?:\.(?:[\p{L}\p{N}\p{M}](?![\p{L}\p{N}\p{M}-]{1,2}--)(?:[\p{L}\p{N}\p{M}-]{0,61}[\p{L}\p{N}\p{M}])?|xn--(?![A-Za-z0-9]{2}--)(?:[A-Za-z0-9-]{1,59}[A-Za-z0-9])?|[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?))+\.?)))(?<![\p{L}\p{N}\p{M}_])(?:(?:(?:https?|ftps?)://(?:[\p{L}\p{N}\p{M}\-._\~!$&'()*+,;=%]+(?::[\p{L}\p{N}\p{M}\-._\~!$&'()*+,;=%]*)?@)?(?:localhost|(?&HOSTNAME)|(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(?:\.(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)){3})|\[[0-9A-Fa-f:]+\]))|(?&HOSTNAME))(?::\d{1,5})?(?:[/?#](?:(?:%[0-9A-Fa-f]{2})|[A-Za-z0-9\-._\~]|[\p{L}\p{N}\p{M}\p{S}\p{P}])*)?(?![\p{L}\p{N}\p{M}_])"


def detect_spam(train):
    compiled = re.compile(stolen_regex, re.IGNORECASE | re.DOTALL)
    for x, y in enumerate(train):
        val = compiled.search(y)
        if val is not None:
            print(x, y)
            print(val, y)
        else:
            print("ahhh")
            print(y)
            print("ahhh")
