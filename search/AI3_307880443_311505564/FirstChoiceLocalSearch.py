'''
@author: Eyal
'''

from abstract_search import LocalSearch
from random import shuffle
from data import load_hw3_data_1
from classifier import KNearestNeighbours
from search.AI3_307880443_311505564.LearningState import create_learning_state_ops, \
    LearningState

class FirstChoiceLocalSearch(LocalSearch):
    def search(self, evaluation_set, evaluation_set_labels, *args, **kwargs):
        current = self._current_state
        current_evaluation = current.evaluate(evaluation_set, evaluation_set_labels, *args, **kwargs)
        
        depth = 0
        while True:
            depth += 1
            print depth
            next_states = current.get_next_states()
            shuffle(next_states)
            for next_state, next_op in next_states:
                next_evaluation = next_state.evaluate(evaluation_set, evaluation_set_labels, *args, **kwargs)
                if next_evaluation > current_evaluation:
                    current = next_state
                    current_evaluation = next_evaluation
                    evaluation_set = next_op.remove_attributes(evaluation_set)
                    self.operators.append(next_op)
                    break
            else:
                print "Finished searching after %d steps" % (depth)
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
    print "State:", state
    print "Ops:", ops
