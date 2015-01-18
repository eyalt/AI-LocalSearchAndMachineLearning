# -*- coding: utf-8 -*-
"""
Created on Thu Jan 01 18:56:35 2015

@author: liorf
"""

def load_hw3_data_1():
    '''Loads the first part data from the data folder.
    all examples are represented as lists of feature values. label is 1 for positive and 0 for negative.
    
    returns: a list containing three tuples: [tuple1, tuple2, tuple3]
    tuple1 is a pair of training data and training labels. 
    tuple2 is a pair of validation set data and the corresponding labels
    tuple3 is a pair of test set and the test set labels. Note that the test set should ONLY be used once!
    '''
    training_data= []
    training_labels= []
    validation_data= []
    validation_labels= []
    test_data= []
    test_labels= []
    
    with open('./data/training1.csv', 'r') as fptr:
        for line in fptr.readlines():
            example= [float(value) for value in line.split(',')[:-1]]
            label= int(line.split(',')[-1])
            training_data.append(example)
            training_labels.append(label)
    with open('./data/validation1.csv', 'r') as fptr:
        for line in fptr.readlines():
            example= [float(value) for value in line.split(',')[:-1]]
            label= int(line.split(',')[-1])
            validation_data.append(example)
            validation_labels.append(label)
    with open('./data/test1.csv', 'r') as fptr:
        for line in fptr.readlines():
            example= [float(value) for value in line.split(',')[:-1]]
            label= int(line.split(',')[-1])
            test_data.append(example)
            test_labels.append(label)
    return [(training_data, training_labels), (validation_data, validation_labels), (test_data, test_labels)]
    
def load_hw3_data_2():
    '''Loads data for cross validation for the second part
    
    Returns a tuple of data and labels
    '''
    data= []
    labels= []
    
    with open('./data/data2.csv', 'r') as fptr:
        for line in fptr.readlines():
            example= [float(value) for value in line.split(',')[:-1]]
            label= int(line.split(',')[-1])
            data.append(example)
            labels.append(label)
    return (data, labels)