#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import sys, getopt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn import cross_validation, metrics

N_JOBS = -1
IID = False
CRITERION = "gini"
OOB_SCORE = True

WEIGHT_LOW = 0.1433
WEIGHT_HIGH = 0.8567
# CLASS_WEIGHT = {1:WEIGHT_LOW, -1:WEIGHT_HIGH}
CLASS_WEIGHT = "balanced"

N_ESTIMATORS = 10
MAX_LEAF_NODES = None
MAX_FEATURES = "sqrt"
MIN_SAMPLES_LEAF = 1
MIN_SAMPLES_SPLIT = 2
MAX_DEPTH = None

X1 = None
y1 = None
X2 = None
y2 = None

def myopt(argv):
    if len(argv) < 2:
        print("Usage: $0 -n traning_file -t testing_file")
        sys.exit(1)
    try:
        opts, args = getopt.getopt(argv[1:], "hn:t:", ["train=", "test="])
    except getopt.GetoptError:
        print("Usage: $0 -n traning_file -t testing_file")
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-h":
            print("Usage: $0 -n traning_file -t testing_file")
            sys.exit()
        elif opt in ("-n", "--train"):
            train_file = arg
        elif opt in ("-t", "--test"):
            test_file = arg
    return train_file, test_file

def sv2csv(file0):
    fin = open(file0, "r")
    fileout = file0 + ".csv"
    fout = open(fileout, "w")
    line = fin.readline()
    line_sep = line.split()
    for m in range(len(line_sep)):
        if 0 == m:
            fout.write("label")
            continue
        fout.write(",f" + str(m))
    fout.write('\n')
    while line:
        if len(line) < 2:
            break
        line_sep = line.split()
        for m in range(len(line_sep)):
            if 0 == m:
                fout.write(line_sep[0])
                continue
            m_sep = line_sep[m].split(':')
            fout.write(',' + m_sep[1])
        fout.write('\n')
        line = fin.readline()
    fin.close()
    fout.close()
    print(fileout)
    return fileout

def grid_param(param_test):
    global N_JOBS
    global IID
    global CRITERION
    global OOB_SCORE
    global WEIGHT_LOW
    global WEIGHT_HIGH
    global CLASS_WEIGHT
    global N_ESTIMATORS
    global MAX_LEAF_NODES
    global MAX_FEATURES
    global MIN_SAMPLES_LEAF
    global MIN_SAMPLES_SPLIT
    global MAX_DEPTH
    global X1
    global y1
    global X2
    global y2

    gsearch = GridSearchCV(estimator = RandomForestClassifier(n_estimators = N_ESTIMATORS, max_depth = MAX_DEPTH,
            max_leaf_nodes = MAX_LEAF_NODES, min_samples_split = MIN_SAMPLES_SPLIT, min_samples_leaf = MIN_SAMPLES_LEAF,
            max_features = MAX_FEATURES, oob_score = OOB_SCORE, class_weight = CLASS_WEIGHT), 
            param_grid = param_test, scoring = "roc_auc", cv = 10, n_jobs = N_JOBS)
    gsearch.fit(X1, y1)

    if gsearch.best_params_.has_key("n_estimators"):
        N_ESTIMATORS = gsearch.best_params_["n_estimators"]
        print("n_estimators: ", N_ESTIMATORS)
    if gsearch.best_params_.has_key("max_depth"):
        MAX_DEPTH = gsearch.best_params_["max_depth"]
        print("max_depth: ", MAX_DEPTH)
    if gsearch.best_params_.has_key("min_samples_split"):
        MIN_SAMPLES_SPLIT = gsearch.best_params_["min_samples_split"]
        print("min_samples_split: ", MIN_SAMPLES_SPLIT)
    if gsearch.best_params_.has_key("min_samples_leaf"):
        MIN_SAMPLES_LEAF = gsearch.best_params_["min_samples_leaf"]
        print("min_samples_leaf: ", MIN_SAMPLES_LEAF)
    if gsearch.best_params_.has_key("max_features"):
        MAX_FEATURES = gsearch.best_params_["max_features"]
        print("max_features: ", MAX_FEATURES)
    if gsearch.best_params_.has_key("max_leaf_nodes"):
        MAX_LEAF_NODES = gsearch.best_params_["max_leaf_nodes"]
        print("max_leaf_nodes: ", MAX_LEAF_NODES)
    return

if __name__ == "__main__":
    train_file, test_file = myopt(sys.argv)
    train_csv = sv2csv(train_file)
    test_csv = sv2csv(test_file)
    train = pd.read_csv(train_csv)
    test = pd.read_csv(test_csv)
    x1_col = [x for x in train.columns if x != "label"]
    X1 = train[x1_col]
    y1 = train["label"]
    x2_col = [x for x in train.columns if x != "label"]
    X2 = train[x2_col]
    y2 = train["label"]

    param_test1 = {"n_estimators":range(10, 151, 10)}
    grid_param(param_test1)
    print("1 n_estimators set")
    param_test2 = {"max_features":range(10, 46, 5)}
    grid_param(param_test2)
    print("2 max_features set")
    param_test3 = {"min_samples_split":range(10, 251, 10), "min_samples_leaf":range(10, 60, 5)}
    grid_param(param_test3)
    print("3 min_samples_split set")
    print("3 min_samples_leaf set")
    param_test4 = {"max_depth":range(6, 15, 2), "min_samples_split":range(10, 251, 10)}
    grid_param(param_test4)
    print("4 min_samples_split set")
    print("4 max_depth set")
    # prama_test5 = {"max_leaf_nodes":range()}
    # grid_param(param_test5)
    # print("5 max_leaf_nodes set")
    param_test6 = {"min_samples_split":range(10, 251, 10), "min_samples_leaf":range(10, 60, 5)}
    grid_param(param_test6)
    print("6 min_samples_split set")
    print("6 min_samples_leaf set")
    param_test7 = {"max_features":range(10, 46, 5)}
    grid_param(param_test7)
    print("7 max_features set")

    rf = RandomForestClassifier(n_estimators = N_ESTIMATORS, max_depth = MAX_DEPTH, max_leaf_nodes = MAX_LEAF_NODES,
        min_samples_split = MIN_SAMPLES_SPLIT, min_samples_leaf = MIN_SAMPLES_LEAF, max_features = MAX_FEATURES,
        oob_score = OOB_SCORE, class_weight = CLASS_WEIGHT, n_jobs = N_JOBS)
    rf.fit(X1, y1)
    print("oob_score_: ", rf.oob_score_)
    y1_predprob = rf.predict_proba(X1)[:,1]
    print(y1_predprob)
    print "AUC Score (Train): %f" % metrics.roc_auc_score(y1, y1_predprob)

    test_class = rf.predict(X2)
    test_proba = rf.predict_proba(X2)
    fout = open(test_csv + ".out", "w")
    for i in range(len(y2)):
        fout.write(str(y2[i]) + '\t' + str(test_proba[i][1]) + '\t' + str(test_class[i]) + '\n')
    fout.close()
    print("ACC: ", rf.score(X2, y2))
