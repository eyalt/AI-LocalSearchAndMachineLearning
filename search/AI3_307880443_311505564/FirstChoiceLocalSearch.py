'''
Created on 19 αιπε 2015

@author: Eyal
'''

from abstract_search import LocalSearch
from random import shuffle

class FirstChoiceLocalSearch(LocalSearch):
    def search(self, evaluation_set, evaluation_set_labels, *args, **kwargs):
        current = self._current_state
        current_evaluation = current.evaluate(evaluation_set, evaluation_set_labels, *args, **kwargs)
        
        while True:
            for next_state, next_op in shuffle(current.get_next_states()):
                next_evaluation = next_state.evaluate(evaluation_set, evaluation_set_labels, *args, **kwargs)
                if next_evaluation > current_evaluation:
                    current = next_state
                    current_evaluation = next_evaluation
                    self.operators.append(next_op)
                    break
            else:
                return current, self.operators
