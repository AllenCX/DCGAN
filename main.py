import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import sys 
import os
from config import Config
from DCGAN import DCGAN
if __name__ == "__main__":
	NUM_GENERATED = 9

	tf.reset_default_graph()
	sess = tf.Session()

	config = Config(
	    batch_size=64, 
	    latent_size=100, 
	    lr=0.0002, 
	    epoch_num=30, 
	    beta1=0.5, 
	    alpha=0.2, 
	    save_per_epoch=5)

	test_input = tf.placeholder(shape=[None, 28, 28, 1], dtype=tf.float32)
	test_model = DCGAN(test_input, tf.Session(), config)
	test_model.train(config)

	generated_images = test_model.generate(NUM_GENERATED)
	tmp = int(np.sqrt(NUM_GENERATED))
	f, axarr = plt.subplots(tmp, tmp, figsize=(5, 5))
	
	for i in range(tmp):
	    for j in range(tmp):
	        axarr[i, j].imshow(generated_images[i*tmp + j], cmap='Greys', interpolation='nearest')

	f.savefig("output.png")