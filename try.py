import layers
import tensorflow_core as tf
import math 
import numpy as np

inputs = [np.zeros((16,8192,3), dtype=np.float32), None]
outputs = layers.pointSIFT_res_module(radius=0.1, out_channel=64, merge='concat')(inputs) 
