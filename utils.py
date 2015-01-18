# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 13:35:35 2014

@author: liorf
"""

from collections import Counter
from math import log, sqrt

def entropy(labels):
    '''Calculate Shannon entropy for set of labels. 
    Note that this works even for non-binary labels.
    This is useful for decision trees.
    
    Returns a float from 0.0 to 1.0 
    '''
    counts= Counter(labels)
    freqs= [float(count)/len(labels) for (value, count) in counts.most_common()]
    nonzero_freqs= [f for f in freqs if f > 0]
    if len(nonzero_freqs) <= 1:
        return 0.0 #edge case
    return sum([-log(f, 2)*f for f in nonzero_freqs])
    
def information_gain(old_labels, feature_values):
    '''Calculate information gain by splitting old_labels according to feature_values
    Note that this works for nominal features as well as binary (but it tends to prefer features with many values. there is a correction known as information gain ratio, but it is beyond the scope of the course)
    
    Returns a float
    '''
    current_entropy= entropy(old_labels)
    
    conditional_entropy= 0.0
    for value in frozenset(feature_values):
        indices_for_value= [i for (i,v) in enumerate(feature_values) if v==value]
        value_probability= float(len(indices_for_value))/len(feature_values)
        
        conditional_labels= [label for (i, label) in enumerate(old_labels) if i in indices_for_value]
        conditional_entropy+= value_probability*entropy(conditional_labels)
        
    return current_entropy - conditional_entropy
    
def l2_distance(example1, example2):
    '''Calculate distance between two examples based on L2-Norm (euclidean distance)
    Note that both examples must be lists of the same length!
    
    Returns a float representing the distance.
    '''
    if len(example1)!=len(example2):
        raise ValueError('cannot calculate distance on vectors of different dimensions')
    return sqrt(sum([(v1-v2)**2 for (v1,v2) in zip(example1,example2)]))
    
SIGNIFICANCE_THRESHOLD= 0.05
def student_paired_t_test(original_measurements, measurements_after_alteration):
    '''This is the paired T-test (for repeated measurements):
    Given two sets of measurements on the SAME data points (folds) before and after some change (In our case, before and after the local search step),
    Checks whether or not they come from the same distribution. 
    
    This T-test assumes the measurements come from normal distributions (if you want, you can use the Mann-Whitney U test (scipy.stats.mannwhitneyu) to check this assumption)
    
    Returns: the probability that the measurements came from the same distribution.
    Note that since we know if the new measurements are better or not, we only want to know the probability 
    For the sake of this assignment, we will say the results are SIGNIFICANT (they truly are different) if this value is less than 0.05.
    Also returns: is_significant = binary value stating whether the result is statically significant. is_better = binary value stating whether the new measurements are better than the old ones.
    '''
    try:
        from scipy.stats import ttest_rel
    except:
        raise Exception('You must either install scipy, or find an online implementation of paired T-test')
    test_value, probability= ttest_rel(original_measurements, measurements_after_alteration)
    is_significant= probability/2 < SIGNIFICANCE_THRESHOLD
    is_better= sum(original_measurements) < sum(measurements_after_alteration) #should actually compare averages, but there's no need since it's the same number of measurments.
    return probability/2 if is_better else 1-probability/2, is_significant, is_better