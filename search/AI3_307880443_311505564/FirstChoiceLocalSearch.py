'''
@author: Eyal
'''

from abstract_search import LocalSearch
from random import shuffle
from data import load_hw3_data_1
from classifier import KNearestNeighbours
from search.AI3_307880443_311505564.LearningState import create_learning_state_ops, \
    LearningState, calc_accuracy
import time

class FirstChoiceLocalSearch(LocalSearch):
    def search(self, evaluation_set, evaluation_set_labels, *args, **kwargs):
        current = self._current_state
        current_evaluation = current.evaluate(evaluation_set, evaluation_set_labels, *args, **kwargs)
        
        depth = 0
        while True:
            depth += 1
            print "Step", depth
            start = time.clock()
            next_states = current.get_next_states()
            print "Generating %d new states. time: %f" % (len(next_states), time.clock() - start)
            shuffle(next_states)
            num_evaluations = 0
            start = time.clock()
            for next_state, next_op in next_states:
                num_evaluations += 1
                next_evaluation_set = next_op.remove_attributes(evaluation_set)
                next_evaluation_set_labels = evaluation_set_labels
                next_evaluation = next_state.evaluate(next_evaluation_set, next_evaluation_set_labels, *args, **kwargs)
                if next_evaluation > current_evaluation:
                    print "Found a better state after %d evaluations. time: %f" % (num_evaluations, time.clock() - start)
                    print "Chosen op:", next_op
                    print "Chosen state:", next_state
                    current = next_state
                    current_evaluation = next_evaluation
                    evaluation_set = next_evaluation_set
                    evaluation_set_labels = next_evaluation_set_labels
                    assert len(current.training_set[0]) == len(evaluation_set[0])
                    self.operators.append(next_op)
                    break
            else:
                print "Finished searching after %d steps. time: %f" % (depth, time.clock() - start)
                return current, self.operators


if __name__ == "__main__":
    training, validation, test = load_hw3_data_1()
    training_set, training_labels = training
    validation_set, validation_labels = validation
    test_set, test_labels = test
    classifier = KNearestNeighbours(3)
    
    startingState = LearningState(create_learning_state_ops(training_set), training_set, training_labels)
    localSearch = FirstChoiceLocalSearch(startingState)
    
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

