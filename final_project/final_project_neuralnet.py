__author__ = 'kosuket'
import sys
import os
import numpy as np
import random


def extract_foldername():
    argvs = sys.argv
    if len(argvs) != 2:
        print u'args foldername required'.format(argvs[0])
        quit()
    afolder = argvs[1]
    return afolder


def get_iteration(folderpath):
    files = os.listdir(folderpath)
    totalfilecnt = 0
    for afile in files:
        file_fullpath = folderpath + "/" + afile
        if file_fullpath[len(file_fullpath) - 4:] == ".csv":
            totalfilecnt += 1
    return totalfilecnt


def create_data(folderpath, testidx, mode="stat"):
    def format_rawdata():
        if class_flg == 0:
            col_mal = np.zeros((rawdata.shape[0]))
        else:
            col_mal = np.ones((rawdata.shape[0]))
        return np.c_[col_mal, rawdata]

    def format_statdata():
        # mean_d
        ave_d = np.average(rawdata[:, 0])
        # mean_s
        ave_s = np.average(rawdata[:, 1])
        # std_d
        var_d = np.var(rawdata[:, 0])
        # std_s
        var_s = np.var(rawdata[:, 1])
        return np.array([[class_flg, var_d * var_s, ave_d ** 2 + ave_s ** 2]])

    def randomize_order(adata):
        for i in range(adata.shape[0] * 2):
            index1 = random.randint(0, adata.shape[0] - 1)
            index2 = random.randint(0, adata.shape[0] - 1)
            adata[index1], adata[index2] = np.array(adata[index2]), np.array(adata[index1])
        return np.concatenate((adata, -np.ones((np.shape(adata)[0], 1))), axis=1)

    # read files in the folder
    files = os.listdir(folderpath)
    filecnt = 0
    trainfilecnt = 0
    train_original_data = np.empty(shape=[0, 0])
    test_original_data = np.empty(shape=[0, 0])
    for afile in files:
        file_fullpath = folderpath + "/" + afile
        if file_fullpath[len(file_fullpath) - 4:] == ".csv":
            class_flg = int(afile[len(afile) - 5])
            rawdata = np.loadtxt(file_fullpath, delimiter=',', skiprows=1, usecols=(1, 2))
            # format data
            newdata = format_rawdata() if mode == "raw" else format_statdata()
            # merge files
            if filecnt != testidx:
                train_original_data = newdata if trainfilecnt == 0 else np.r_[train_original_data, newdata]
                trainfilecnt += 1
            else:
                test_original_data = newdata
            filecnt += 1
    if trainfilecnt < 1:
        print "no file in the folder"
        quit()
    # randomize the order and return
    return randomize_order(train_original_data), randomize_order(test_original_data)


def separate_class_and_input(adata):
    aclass = adata[:, 0]
    aclass = aclass.reshape((np.shape(aclass)[0], 1))
    ainput = adata[:, 1:]
    return aclass, ainput


def train_slp(input_t, target, mu, itn):
    # initialize
    w = np.random.rand(np.shape(input_t)[1], np.shape(target)[1]) * 0.1 - 0.05

    # train
    for n in range(itn):
        y = execute_slp(input_t, w)
        w = w - mu * np.dot(np.transpose(input_t), y - target)
    return w


def execute_slp(input_e, weight):
    output = np.dot(input_e, weight)
    output = np.where(output > 0, 1, 0)
    return output


def train_mlp(input, target, mu, iteration):
    # initialize
    input_cnt = np.shape(input)[1]
    output_cnt = np.shape(target)[1]
    hidden_cnt = input.shape[1]
    w_input = (np.random.rand(input_cnt, hidden_cnt) - 0.5) * 2 / np.sqrt(input_cnt)
    w_hidden = (np.random.rand(hidden_cnt + 1, output_cnt) - 0.5) * 2 / np.sqrt(hidden_cnt)
    new_weight1 = np.zeros((np.shape(w_input)))
    new_weight2 = np.zeros((np.shape(w_hidden)))

    # train
    for n in range(iteration):
        output, hidden = execute_mlp(input, w_input, w_hidden, "train")
        d_output = (output - target) * output * (1.0 - output)
        d_hidden = hidden * 0.9 * (1.0 - hidden) * (np.dot(d_output, np.transpose(w_hidden)))
        new_weight1 = mu * (np.dot(np.transpose(input), d_hidden[:, :-1])) + 0.9 * new_weight1
        new_weight2 = mu * (np.dot(np.transpose(hidden), d_output)) + 0.9 * new_weight2
        w_input -= new_weight1
        w_hidden -= new_weight2
    return w_input, w_hidden


def execute_mlp(input, w_input, w_hidden, mode="train"):
    hidden = np.dot(input, w_input)
    hidden = 1.0 / (1.0 + 0.9 * np.exp(-hidden))
    hidden = np.concatenate((hidden, -np.ones((np.shape(input)[0], 1))), axis=1)
    output = np.dot(hidden, w_hidden)
    return 1.0 / (1.0 + np.exp(-output)), hidden if mode == "train" else 1.0 / (1.0 + np.exp(-output))


def output_confusion_matrix(output, target):
    output = np.where(output > 0.5, 1, 0)
    confusion_matrix = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            confusion_matrix[i, j] = np.sum(np.where(output == i, 1, 0) * np.where(target == j, 1, 0))


def calculate_metrics(output, target, mode="partial_result"):
    output = np.where(output > 0.5, 1, 0)
    confusion_matrix = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            confusion_matrix[i, j] = np.sum(np.where(output == i, 1, 0) * np.where(target == j, 1, 0))
    accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix) * 100 if np.sum(confusion_matrix) > 0 else 0
    precision = confusion_matrix[0, 0] / np.sum(confusion_matrix[:, 0]) * 100 if np.sum(confusion_matrix[:, 0]) > 0 else 0
    recall = confusion_matrix[0, 0] / np.sum(confusion_matrix[0, :]) * 100 if np.sum(confusion_matrix[0, :]) > 0 else 0
    sensitivity = confusion_matrix[0, 0] / np.sum(confusion_matrix[0, :]) * 100 if np.sum(confusion_matrix[0, :]) > 0 else 0
    specificity = confusion_matrix[1, 1] / np.sum(confusion_matrix[1, :]) * 100 if np.sum(confusion_matrix[1, :]) > 0 else 0
    f_measure = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    return accuracy, precision, recall, sensitivity, specificity, f_measure, confusion_matrix if mode == "full_result" else accuracy

# noinspection PyUnboundLocalVariable
def roc_curve(output, target):
    roc_data = np.c_[output, target]
    indices = np.argsort(roc_data[:, 0])
    roc_data = roc_data[indices, :]
    for i in range(100):
        pre = random.randint(0, (roc_data.shape[0] - 1) / 2)
        post = pre + random.randint(0, (roc_data.shape[0] - 1) / 2)
        acrcy, prcsn, rcl, snstvty, spcfcty, fms, cnfsnmtrx = calculate_metrics(roc_data[pre:post, 0],
                                                                                roc_data[pre:post, 1], "full_result")
        if i == 0:
            roc_result = np.array([[100 - spcfcty, snstvty]])
        else:
            roc_result = np.r_[roc_result, np.array([[100 - spcfcty, snstvty]])]
    idx = np.argsort(roc_result[:, 0])
    roc_result = roc_result[idx, :]
    return roc_result


def output_result():
    if multi_flg:
        print "Mode: Neural Network MLP"
    else:
       print "Mode: Neural Network SLP"
    print "Average Accuracy = ", acrcy
    print "Average Precision = ", prcsn
    print "Average Recall = ", rcl
    print "Average Sensitivity = ", snstvty
    print "Average Specificity = ", spcfcty
    print "Average F measure = ", fms
    print "Average Confusion Matrix = ", cnfsnmtrx
    print "ROC Curve:", roc_result


# main
folder = extract_foldername()
iteration = get_iteration(folder)
multi_flg = False
all_output = np.empty(shape=[0, 0])
all_target = np.empty(shape=[0, 0])

for i in range(iteration):
    # take i-th data for test
    train, test = create_data(folder, i, "stat")
    train_class, train_data = separate_class_and_input(train)
    test_class, test_data = separate_class_and_input(test)
    if multi_flg:  # If true, execute Multilayer Perceptron. If not, Single Layer Perceptron
        trained_wi, trained_wh = train_mlp(train_data, train_class, 0.1, 10000)
        ys = execute_mlp(test_data, trained_wi, trained_wh, "test")
    else:
        trained_w = train_slp(train_data, train_class, 0.01, 10000)
        ys = execute_slp(test_data, trained_w)
    if i == 0:
        all_output = ys
        all_target = test_class
    else:
        all_output = np.r_[all_output, ys]
        all_target = np.r_[all_target, test_class]
acrcy, prcsn, rcl, snstvty, spcfcty, fms, cnfsnmtrx = calculate_metrics(all_output, all_target, "full_result")
roc_result = roc_curve(all_output, all_target)
output_result()