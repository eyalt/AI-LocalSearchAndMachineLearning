'''
Created on 19 αιπε 2015

@author: Eyal
'''
from abstract_search import SearchState

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
    def __init__(self,rows_to_remove,cols_to_remove):
        self._rows_to_remove = rows_to_remove
        self._cols_to_remove = cols_to_remove
        
    def __call__(self, learningState):
        ops = create_learning_state_ops(learningState.training_set)
        
def create_learning_state_ops(subjects_set):
    pass