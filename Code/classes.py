"""
Author: Martin Schiemer
Classes for saving different parameters used to create the information plane.
"""


class Weights():
    def __init__(self):
        self.name = "weights"
        self.act_func = []
        self.dic = {}
        self.flat_weights = {}
        self.max = {}
        self.min = {}
        self.gradients = {}
        self.digitized = {}
        self.batchsize = 0
        self.int_model_score = {}
        self.model_score = 0
        

class Outputs:
    def __init__(self):
        self.name = "outputs"
        self.act_func = []
        self.dic = {}
        self.flat_outputs = {}
        self.max = {}
        self.min = {}
        self.gradients = {}
        self.digitized = {}
        self.batchsize = 0
        self.int_model_score = {}
        self.model_score = 0
        
class Binning_MI():
    def __init__(self):
        self.name = "Binning_MI"
        self.act_func = []
        self.px = []
        self.py = []
        self.tot_bins = {}
        self.bin_edges = {}
        self.unused_bins = {}
        self.inverse_x = []
        self.inverse_y = []
        self.py_x = []
        self.p_layer_act = {}
        self.p_layer_w = {}
        self.w_entropy_dic = {}
        self.act_entropy_dic = {}
        self.mi_x_y = 0
        self.entropy_x = 0
        self.mi_x = {}
        self.mi_y = {}
        self.batchsize = 0
        self.model_score = 0

class KDE_MI():
    def __init__(self):
        self.name = "KDE_MI"
        self.mi_x = {}
        self.mi_y = {}
        self.act_func = []
        self.batchsize = 0
        self.model_score = 0

class EDGE_MI():
    def __init__(self):
        self.name = "EDGE_MI"
        self.mi_x = {}
        self.mi_y = {}
        self.act_func = []
        self.batchsize = 0
        self.model_score = 0
        
class Kras_MI():
    def __init__(self):
        self.name = "Kras_MI"
        self.mi_x = {}
        self.mi_y = {}
        self.act_func = []
        self.batchsize = 0
        self.model_score = 0