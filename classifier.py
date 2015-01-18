# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 13:35:35 2014

@author: liorf
"""

import utils
from collections import Counter

class Classifier(object):
    def __init__(self):
        '''Method for initializing'''
        raise NotImplementedError('Abstract class, do not instantiate')
        
    def train(self, training_set, training_set_labels):
        '''Method for training the classifier'.
        training_set is a list of examples (an example is a list of feature values).
        training_set_labels is a list of labels for those examples.
        
        returns None.
        '''
        raise NotImplementedError
        
    def classify(self, test_set):
        '''Method for classifying a test set.
        test_set is a list of examples (an example is a list of feature values)
        
        returns the predicted labels (according to the classifier) of the examples given as input
        '''
        raise NotImplementedError
        
        
class KNearestNeighbours(Classifier):
    def __init__(self, k=1):
        '''
        parameter k is the number of neighbours considered when deciding on a label. It should be odd and >= 1.
        '''
        self.__k= k
        
    def train(self, training_set, training_set_labels):
        if len(training_set) < self.__k: #edge case
            raise ValueError('Please give a sufficiently large training set!')
        
        #lazy/naive implementation. there is a more efficient way to train, but it is beyond the scope of the course
        self.examples= training_set
        self.__labels= training_set_labels
        
    def classify(self, test_set):
        test_labels= []
        
        for example in test_set:
            #calculate euclidian distances
            distances= [utils.l2_distance(example, training_example) for training_example in self.examples]
            #sort indexes of training set by nearest to example
            sorted_neighbour_indices= [i for (distance, i) in sorted((distance, i) for (i, distance) in enumerate(distances))]
            #take first k labels
            neighbour_labels= [self.__labels[i] for i in sorted_neighbour_indices[:self.__k]]
            #decide by majority
            counts= Counter(neighbour_labels)
            (most_common_label, apperance_count)= counts.most_common(1)[0] #see documentation
            test_labels.append(most_common_label)
            
        return test_labels

class DecisionTree(Classifier):
    '''Class implementing decision tree classifier.
    
    Uses a common library (sklearn) due to efficiency concerns.
    '''
    def __init__(self, min_leaf_size=1):
        '''Decision tree with optional pre-pruning.
        if min_leaf_size is more than 1, pre-pruning occurs
        '''
        try:
            from sklearn import tree
        except:
            raise NotImplementedError('Download sklearn, or replace this implementation with another')
        self.classifier= tree.DecisionTreeClassifier(criterion='entropy', min_samples_leaf=min_leaf_size, random_state=0)
        
    def train(self, training_set, training_set_labels):
        
        self.classifier.fit(training_set, training_set_labels)
        return
        
    def classify(self, test_set):
        return self.classifier.predict(test_set)
