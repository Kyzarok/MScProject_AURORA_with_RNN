import numpy as np

import tensorflow as tf
import scipy.io
import matplotlib as mpl
mpl.use('Agg')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


from ae import AE
            
if __name__ == '__main__':
    
    # Variables
    ae = AE()

    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,log_device_placement=False)) as sess:
        init_all_vars_op = tf.variables_initializer(tf.global_variables(), name='init_all_vars_op')
        sess.run(init_all_vars_op)        
        tf.train.write_graph(sess.graph_def, '../resources', 'graph.pb', as_text=False)
        save_path_init = ae.saver.save(sess, "../resources/model_init.ckpt")
