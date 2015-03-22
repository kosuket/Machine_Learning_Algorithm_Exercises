__author__ = 'kosuket'
import numpy as np
import sys
import os
import random
import math


def extract_args():
    k = 1  # default k = 1
    argvs = sys.argv
    if len(argvs) == 3:
        afolder = argvs[1]
        k = int(argvs[2])
    elif len(argvs) == 2:
        afolder = argvs[1]
    else:
        print 'args foldername and k required'
        quit()
    return afolder,k


def get_iteration(folderpath):
    files = os.listdir(folderpath)
    totalfilecnt = 0
    for afile in files:
        file_fullpath = folderpath + "/" + afile
        if file_fullpath[len(file_fullpath) - 4:] == ".csv":
            totalfilecnt += 1
    return totalfilecnt


def create_data(folderpath, testidx, mode="raw"):
    def format_rawdata():
        if class_flg == 0:
            col_mal = np.zeros((rawdata.shape[0]))
        else:
            col_mal = np.ones((rawdata.shape[0]))
        return np.c_[col_mal, rawdata]

    def format_statdata():
        sum_dif_d = 0
        sum_dif_s = 0
        sum_tend_d = 0
        sum_tend_s = 0
        for i in range(rawdata.shape[0] - 2):
            sum_dif_d += math.fabs(rawdata[i, 0] - rawdata[i + 1, 0])
            sum_dif_s += math.fabs(rawdata[i, 1] - rawdata[i + 1, 1])
            if i < rawdata.shape[0] - 2:
                nmd1 = rawdata[i, 0] - rawdata[i + 1, 0]
                nmd2 = rawdata[i + 1, 0] - rawdata[i + 2, 0]
                nms1 = rawdata[i, 1] - rawdata[i + 1, 1]
                nms2 = rawdata[i + 1, 1] - rawdata[i + 2, 1]
                if nmd1 * nmd2 != 0:
                    if nmd1 / math.fabs(nmd1) != nmd2 / math.fabs(nmd2):
                        sum_tend_d += 1
                if nms1 * nms2 != 0:
                    if nms1 / math.fabs(nms1) != nms2 / math.fabs(nms2):
                        sum_tend_s += 1
        var_d = np.var(rawdata[:, 0])
        var_s = np.var(rawdata[:, 1])
        return np.array([[class_flg, sum_dif_d, sum_dif_s, sum_tend_d, sum_tend_s, math.sqrt(var_d * var_s)]])

    def randomize_order(adata):
        for i in range(adata.shape[0] * 2):
            index1 = random.randint(0, adata.shape[0] - 1)
            index2 = random.randint(0, adata.shape[0] - 1)
            adata[index1], adata[index2] = np.array(adata[index2]), np.array(adata[index1])
        return np.concatenate((adata, -np.ones((np.shape(adata)[0], 1))), axis=1)

    def separate_class_and_input(adata):
        aclass = adata[:, 0]
        aclass = aclass.reshape((np.shape(aclass)[0], 1))
        ainput = adata[:, 1:]
        return aclass, ainput

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
    if filecnt < 1:
        print "no file in the folder"
        quit()
    train_original_data = randomize_order(train_original_data)
    test_original_data = randomize_order(test_original_data)
    train_class, train_data = separate_class_and_input(train_original_data)
    test_class, test_data = separate_class_and_input(test_original_data)
    return train_data, train_class, test_data, test_class


def execute_knn(adata, adata_class, ainput, ak):
    # initialize
    input_cnt = np.shape(ainput)[0]
    nearest = np.zeros(input_cnt)
    # find k nearest neighbors
    for n in range(input_cnt):
        distance = np.sum((adata-ainput[n, :]) ** 2, axis=1)
        index = np.argsort(distance, axis=0)
        potential_class = np.unique(adata_class[index[:k]])
        if len(potential_class) == 1:
            nearest[n] = np.unique(potential_class)
        else:
            majority = np.zeros(max(potential_class) + 1)
            for i in range(ak):
                idx = int(adata_class[index[i]])
                majority[idx] += 1.0 * 1 / (i + 1)
            nearest[n] = potential_class[0] if majority[0] > majority[1] else potential_class[1]
    return nearest


def calc_metrics(output, target):
    output = np.where(output > 0.5, 1, 0)
    confusion_matrix = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            confusion_matrix[i, j] = np.sum(np.where(output == i, 1, 0) * np.where(target == j, 1, 0))
    accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix) * 100 if np.sum(confusion_matrix) > 0 else 0
    precision = confusion_matrix[0, 0] / np.sum(confusion_matrix[:, 0]) * 100 if np.sum(confusion_matrix[:, 0]) > 0 else 0
    recall = confusion_matrix[0, 0] / np.sum(confusion_matrix[0, :]) * 100 if np.sum(confusion_matrix[0, :]) else 0
    sensitivity = confusion_matrix[0, 0] / np.sum(confusion_matrix[0, :]) * 100 if np.sum(confusion_matrix[0, :]) else 0
    specificity = confusion_matrix[1, 1] / np.sum(confusion_matrix[1, :]) * 100 if np.sum(confusion_matrix[1, :]) else 0
    f_measure = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    return accuracy, precision, recall, sensitivity, specificity, f_measure, confusion_matrix


def output_result():
    print "mode: ", k, "- Nearest Neighbors"
    print "Average Accuracy = ", acrcy
    print "Average Precision = ", prcsn
    print "Average Recall = ", rcl
    print "Average Sensitivity = ", snstvty
    print "Average Specificity = ", spcfcty
    print "Average F measure = ", fms
    print "Average Confusion Matrix = ", cnfsnmtrx


# main
folder, k = extract_args()
iteration = get_iteration(folder)
all_output = np.empty(shape=[0, 0])
all_target = np.empty(shape=[0, 0])
for i in range(iteration):
    data, data_class, test, testt = create_data(folder, i, "stat")
    testy = execute_knn(data, data_class, test, k)
    testy = testy.reshape((np.shape(testy)[0], 1))
    if i == 0:
        all_output = testy
        all_target = testt
    else:
        all_output = np.r_[all_output, testy]
        all_target = np.r_[all_target, testt]
acrcy, prcsn, rcl, snstvty, spcfcty, fms, cnfsnmtrx = calc_metrics(all_output, all_target)
output_result()