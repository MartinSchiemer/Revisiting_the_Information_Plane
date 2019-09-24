import unittest
import numpy as np
np.random.seed(1337)
import math
import multiprocessing
import re
from joblib import Parallel, delayed

NUM_CORES = multiprocessing.cpu_count()


import binning
import classes


class Test_Prob_Calc(unittest.TestCase):
    
    def setUp(self):
        self.activations = classes.Outputs()
        self.activations.dic = {(0, 0): [np.array([[-0.1, -0.2, -0.5], [0.6, 0.7, 0.8],
                                                   [0.6, 0.7, 0.1]])],
                                (0, 1): [np.array([[0.15, 0.25, 0.55], [0.65, 0.75, 0.85]])],
                                (1, 0): [np.array([[0.1, 0.2, 0.5], [0.6, 0.7, 0.8]])],
                                (1, 1): [np.array([[0.15, 0.25, 0.45], [0.55, 0.75, 0.55]])],
                                         }
        self.activations.max = {0: 0.82, 1: 1}
        self.activations.min = {0: 0.1, 1: 0.15}
        self.probs = classes.Probabilities()
        self.x = np.array([[0.1,0.1,0.2,0.1,0.3,0.5,0.15],
             [0.14,0.16,0.2,0.7,0.1,0.2,0.4]])
        self.y = np.array([[0,1],
             [1,0]])
        self.bin_amount = 2
        self.bin_size_or_nr = False
    
    def tearDown(self):
        pass
    
    def test_find_max_min_activation(self):
        
        self.activations.max, self.activations.min = \
            probability_calc.find_max_min_activation(self.activations.dic)
        
        self.assertEqual(self.activations.max[0], 0.8)
        self.assertEqual(self.activations.min[0], -0.5)
        self.assertEqual(self.activations.max[1], 0.85)
        self.assertEqual(self.activations.min[1], 0.15)
        
        
    def test_calc_bins(self):
        min_v = 0.1
        max_v = 2
        
        self.assertListEqual(probability_calc.calc_bins("tanh",
                                                     self.bin_size_or_nr, self.bin_amount,
                                                     max_v, min_v).tolist(), [-1.0,  0.0,  1.0001])
        self.assertListEqual(probability_calc.calc_bins("sigmoid",
                                                     self.bin_size_or_nr, self.bin_amount,
                                                     max_v, min_v).tolist(), [0.0,  0.5,  1.0001])
        self.assertListEqual(probability_calc.calc_bins("relu",
                                                     self.bin_size_or_nr, self.bin_amount,
                                                     max_v, min_v).tolist(), [0.0,  1.0,  2.0001])
        self.assertListEqual(probability_calc.calc_bins("softmax",
                                                     self.bin_size_or_nr, self.bin_amount,
                                                     max_v, min_v).tolist(), [0.0,  0.5,  1.0001])
    
    
    def test_extract_inout_probs(self):    
        features = np.array([[0,1,1,1],[1,1,0,1],[1,0,1,1],[1,1,1,0],[0,1,1,1]])
        output = np.array([[0,1],[1,0],[1,0],[0,1],[0,1]])
        self.assertListEqual(probability_calc.extract_inout_probs(features, output)[2].tolist(),
                         [0.4,0.2,0.2,0.2])
        self.assertListEqual(probability_calc.extract_inout_probs(features, output)[3].tolist(),
                         [0.6,0.4])
    
    
    def test_calc_act_probs(self):
        result = {(0,0): [np.array((2/3 ,1/3)),"tanh"],
                  (0,1): [np.array((0.5,0.5)),"relu"],
                  (1,0): [np.array((1.0)),"tanh"],
                  (1,1): [np.array((0.5,0.5)),"relu"]}
        
        activation_name_list = ["tanh", "relu"]
        
        self.assertListEqual(probability_calc.calc_act_probs(self.activations,
                                                             self.bin_size_or_nr, self.bin_amount,
                                                             activation_name_list)[0][(0,0)][0].tolist(),
                                                             result[(0,0)][0].tolist()
                                                             )
        self.assertListEqual(probability_calc.calc_act_probs(self.activations,
                                                             self.bin_size_or_nr, self.bin_amount,
                                                             activation_name_list)[0][(1,1)][0].tolist(),
                                                             result[(1,1)][0].tolist()
                                                             )
        
        
if __name__ == "__main__":
    unittest.main()