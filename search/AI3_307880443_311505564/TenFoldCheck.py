'''
@author: Itay
'''

from abstract_search import LocalSearch
from random import shuffle
from data import load_hw3_data_2
from classifier import KNearestNeighbours
from utils import student_paired_t_test
from search.AI3_307880443_311505564.LearningState import create_learning_state_ops, \
    LearningState, calc_accuracy
import time

def build_fold_sets(data,labels,i):
    test_set = [x for x,k in enumerate(data) if k%10 == i]
    test_labels = [x for x,k in enumerate(labels) if k%10 == i]
    validation_set = [x for x,k in enumerate(data) if k%10 == (i+1)%10]
    validation_labels = [x for x,k in enumerate(labels) if k%10 == (i+1)%10]
    training_set = [x for x,k in enumerate(data) if k%10 not in [i,(i+1)%10]]
    test_labels = [x for x,k in enumerate(labels) if k%10 not in [i,(i+1)%10]]
    return [(training_set,training_labels),(test_set,test_labels),(validation_set,validation_labels)]

def test_ten_fold_validation(classifier):
    data, labels = load_hw3_data_2()
    without_local_search_acc = []
    with_local_search_acc = []
    for i in xrange(10):
        training, validation, test = build_fold_sets(data,labels,i)
        training_set, training_labels = training
        validation_set, validation_labels = validation
        test_set, test_labels = test

        # Calculate the accuracy with the local search
        startingState = LearningState(create_learning_state_ops(training_set), training_set, training_labels)
        localSearch = FirstChoiceLocalSearch(startingState)
        state, ops = localSearch.search(validation_set, validation_labels, classifier)
        classifier.train(state.training_set, state.training_labels)
        new_test_set = test_set
        for op in ops:
            new_test_set = op.remove_attributes(new_test_set)
        with_local_search_acc.append(calc_accuracy(classifier.classify(new_test_set), test_labels))

        # Calculate the accuracy without the local search
        classifier.train(training_set, training_labels)
        without_local_search_acc.append(calc_accuracy(classifier.classify(test_set), test_labels))

    probability, is_significant, is_better = student_paired_t_test(without_local_search_acc,with_local_search_acc)
    acc_without_lr = sum(without_local_search_acc)/float(len(without_local_search_acc))
    acc_with_lr = sum(with_local_search_acc)/float(len(with_local_search_acc))
    return acc_without_lr,acc_with_lr,is_significant,is_better


if __name__ == "__main__":
    k1,k2 = 3,5
    l1,l2 = 1,3

    print "KNearestNeighbours testing:"
    print "For K =",k1
    acc_without_lr,acc_with_lr,is_significant,is_better = test_ten_fold_validation(KNearestNeighbours(k1))
    print  "Accuracy without LR:",acc_without_lr,"Accuracy with LR:",acc_with_lr,"Significant:",is_significant,"Better:",is_better
    print "For K =",k2
    acc_without_lr,acc_with_lr,is_significant,is_better = test_ten_fold_validation(KNearestNeighbours(k2))
    print  "Accuracy without LR:",acc_without_lr,"Accuracy with LR:",acc_with_lr,"Significant:",is_significant,"Better:",is_better
    
    print "DecisionTree testing:"
    print "For min leaf size =",l1
    acc_without_lr,acc_with_lr,is_significant,is_better = test_ten_fold_validation(DecisionTree(l1))
    print  "Accuracy without LR:",acc_without_lr,"Accuracy with LR:",acc_with_lr,"Significant:",is_significant,"Better:",is_better
    print "For min leaf size =",l2
    acc_without_lr,acc_with_lr,is_significant,is_better = test_ten_fold_validation(DecisionTree(l2))
    print  "Accuracy without LR:",acc_without_lr,"Accuracy with LR:",acc_with_lr,"Significant:",is_significant,"Better:",is_better
   
