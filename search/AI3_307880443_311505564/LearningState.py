'''
@author: Eyal
'''

from abstract_search import SearchState
import copy
from random import shuffle

class LearningState(SearchState):
    def __init__(self, legal_operators, training_set, training_labels):
        super(LearningState, self).__init__(self, legal_operators)
        self.training_set = training_set
        self.training_labels = training_labels
        
    def evaluate(self, evaluation_set, evaluation_set_labels, classifier, *args, **kwargs):
        classifier.train(self.training_set, self.training_labels)
        classified_labels = classifier.classify(evaluation_set)
        return calc_accuracy(classified_labels, evaluation_set_labels)

def calc_accuracy(classified_labels, real_labels):
    total_tests = len(real_labels)
    return float(len([i for i in range(total_tests) if classified_labels[i] == real_labels[i]])) / total_tests

class LearningStateOperator:
    def __init__(self, rows_to_remove=[], cols_to_remove=[]):
        assert len(rows_to_remove) == 0 or len(cols_to_remove) == 0
        self._rows_to_remove = rows_to_remove
        self._cols_to_remove = cols_to_remove
        
    def __call__(self, learningState):
        new_training_set, new_training_labels = self._update_training(learningState.training_set, learningState.training_labels)
            
        # now the training set and training labels have been updated
        # build the new operations for the new state
        ops = create_learning_state_ops(new_training_set)
        
        return LearningState(ops, new_training_set, new_training_labels)
        
    def remove_attributes(self, example_set):
        assert len(self._cols_to_remove) == 0 or len(self._cols_to_remove) == 1
        
        new_example_set = copy.deepcopy(example_set)
        
        for i in sorted(self._cols_to_remove, reverse=True):
            for example in new_example_set:
                example.pop(i)
        return new_example_set
    
    def _update_training(self, training_set, training_labels):
        '''removes the examples from both the set and the labels, and removes the attributes
        returns a copy of the lists'''
        assert len(self._rows_to_remove) == 0 or len(set(self._rows_to_remove)) == 5 or len(training_set) <= 5
        assert len(self._cols_to_remove) == 0 or len(self._cols_to_remove) == 1
        assert len(self._rows_to_remove) == 0 or len(self._cols_to_remove) == 0
        
        # first copy the set and labels, while removing unnecessary attributes
        new_training_set = self.remove_attributes(training_set)
        new_training_labels = copy.deepcopy(training_labels)
        
        # remove unnecessary examples
        for i in sorted(self._rows_to_remove, reverse=True):
            new_training_set.pop(i)
            new_training_labels.pop(i)
        return new_training_set, new_training_labels
        
def create_learning_state_ops(example_set):
    num_atts = len(example_set[0])
    rows_indexes = range(len(example_set))
    print "Creating learning state ops"
    print "#examples: %d, #atts: %d" % (num_atts, len(example_set))
    
    ops = [LearningStateOperator(cols_to_remove=[i]) for i in range(num_atts)]
    for i in range(num_atts):
        shuffle(rows_indexes)
        ops.append(LearningStateOperator(rows_to_remove=rows_indexes[:5]))
    
    assert len(ops) == 2 * num_atts
    return ops
