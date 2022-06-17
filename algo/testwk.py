#!/usr/bin/env python

import unittest
from clustering import wkmeans
import numpy as np
from sklearn.metrics.cluster import adjusted_rand_score

class TestKMeans(unittest.TestCase):
    
    def setUp(self):
        rng = np.random.default_rng(0)
        x1 = rng.normal(0,1,size = (25,10))
        x2 = rng.normal(3,3,size = (25,10))
        self.X = np.concatenate([x1,x2])
        self.centers, self.labels = wkmeans(self.X,k=2,p=1,threshold = 1e-8) 
        # For this dataset, we can pick an aggressive threshold since it is well separated 
        
        if (self.labels[0] == 1):
            self.labels = 1 - self.labels
            self.centers = self.centers[::-1]
    
    def test_label(self):
        true_labels = np.ones(50,dtype = int)
        true_labels[:25] = 0
        true_labels[38] = 0
        self.assertTrue((self.labels == true_labels).all(),msg = 'Adj Rand Score is {}'.format(adjusted_rand_score(self.labels,true_labels)))
        # If there are necessary adjustments, we would hope the score > 90%
        
    def test_barycenter(self):
        
        center_0 = np.median(np.sort(np.vstack([self.X[:25],self.X[38]]),axis = 1),axis = 0)
        center_1 = np.median(np.sort(np.vstack([self.X[25:38],self.X[39:]]),axis = 1),axis = 0)
        diff_sum_0 = (center_0 - self.centers[0]).sum()
        diff_sum_1 = (center_1 - self.centers[1]).sum()
        
        self.assertTrue((diff_sum_0 < 1e-3) & (diff_sum_1 < 1e-3),
                        msg = 'Sum of Error is: {} and {}'.format(diff_sum_0,diff_sum_1))

if __name__ == '__main__':
    unittest.main()
   
