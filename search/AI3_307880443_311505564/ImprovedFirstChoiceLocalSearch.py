'''
@author: Itay
'''

from abstract_search import LocalSearch
from random import shuffle
from data import load_hw3_data_1
from classifier import KNearestNeighbours
from search.AI3_307880443_311505564.LearningState import create_learning_state_ops, \
    LearningState, calc_accuracy
import time

MAX_RESTARTS = 5

#we now support random restart!
class ImprovedFirstChoiceLocalSearch(FirstChoiceLocalSearch):
    def search(self, evaluation_set, evaluation_set_labels, *args, **kwargs):    
        max_accuracy = 0
        for attempt in xrange(MAX_RESTARTS):
    
            state, ops = super().search(evaluation_set, evaluation_set_labels, classifier)
            for op in ops:
                new_evaluation_set = op.remove_attributes(new_evaluation_set)
            curr_accuracy = state.evaluate(new_evaluation_set, evaluation_set_labels, *args, **kwargs)
            if curr_accuracy>max_accuracy:
                best_state,best_ops = state,ops

        return best_state,best_ops


if __name__ == "__main__":
    training, validation, test = load_hw3_data_1()
    training_set, training_labels = training
    validation_set, validation_labels = validation
    test_set, test_labels = test
    classifier = KNearestNeighbours(3)
    
    startingState = LearningState(create_learning_state_ops(training_set), training_set, training_labels)
    localSearch = ImprovedFirstChoiceLocalSearch(startingState)
    
    state, ops = localSearch.search(validation_set, validation_labels, classifier)
    
    print "Starting state (Hoping didn't change):", startingState
    
    classifier.train(training_set, training_labels)
    print "Before search accuracy:", calc_accuracy(classifier.classify(test_set), test_labels)
    
    print "Final State:", state
    print "Ops:", ops

    classifier.train(state.training_set, state.training_labels)
    new_test_set = test_set
    for op in ops:
        new_test_set = op.remove_attributes(new_test_set)
    print "After search accuracy:", calc_accuracy(classifier.classify(new_test_set), test_labels)

