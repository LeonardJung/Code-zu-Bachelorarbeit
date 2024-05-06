#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from acme.tf.networks import distributions
import random
import sys
sys.path.append('../')
from probability import wasserstein_dist
from probability import energy_dist

# Create a new parameter vector with random entries. Note that these values are not yet probabilities, they have to be put through the softmax function.
class prob_distr(tf.Module):
    def __init__(self,num_probs):
        super().__init__()
        self.p = tf.Variable(tf.random.normal([1,num_probs]),trainable=True)

    def __call__(self):
        return self.p

# The chosen divergence. E stands for Energy distance, W for Wasserstein metric and KL for KL-divergence.
Divergence = 'E'
optim = tf.keras.optimizers.Adam(0.0001)
support = tf.cast(tf.linspace(0,1,3),tf.float32)

# The distribution that we want to approximate
realdist = tf.constant([[0.5,0,0.5]],tf.float32)
dreal = distributions.DiscreteValuedDistribution(
    values=support,
    probs=realdist,
    )

loss = 1
it = 0

for exec in range(10):
    it = 0
    
    # Parameters (in this case equal to output because we have no input) of the Z-network
    dist = prob_distr(3)
    
    # Initialization of arrays for writing data later
    metrics = np.array([])
    c_distributions = np.array([])
    while it < 100000:
        with tf.GradientTape() as t:

            # Create probability distribution from Z-network output
            d1 = distributions.DiscreteValuedDistribution(
                    values=support,
                    probs=tf.nn.softmax(dist()),
                    )

            # Sample a random variable following the distribution we want to approximate and create empirical distribution
            if bool(random.getrandbits(1)):
                tdist = tf.constant([[1,0,0]],tf.float32)
            else:
                tdist = tf.constant([[0,0,1]],tf.float32)
            d2 = distributions.DiscreteValuedDistribution(
                    values=support,
                    probs=tdist,
                    )

            # All possible losses between empirical distribution and output of Z-network are calculated
            losse0 = energy_dist(d1,d2,exponent=2.0)
            lossw0 = wasserstein_dist(d1,d2,exponent=1.0)
            lossk0 = tfp.distributions.kl_divergence(d2,d1)
            losse = tf.reduce_mean(losse0)
            lossw = tf.reduce_mean(lossw0)
            lossk = tf.reduce_mean(lossk0)

            if Divergence == 'E':
                loss = losse
            elif Divergence == 'W':
                loss = lossw
            else:
                loss = lossk

            # All possible losses between real distribution and output of Z-network
            lossreale0 = energy_dist(d1,dreal,exponent=2.0)
            lossrealw0 = wasserstein_dist(d1,dreal,exponent=1.0)
            lossrealk0 = tfp.distributions.kl_divergence(dreal,d1)
            lossreale = tf.reduce_mean(lossreale0)
            lossrealw = tf.reduce_mean(lossrealw0)
            lossrealk = tf.reduce_mean(lossrealk0)

        # actual training step 
        tv = dist.trainable_variables
        grads = t.gradient(loss,tv)
        optim.apply_gradients(zip(grads,tv))

        # save final learned distribution
        with open('data/final_distr_50_0_50_' + Divergence + '_' + str(exec) + '.npy', 'wb') as f:
            np.save(f, tf.nn.softmax(dist()).numpy()[0])

        # construct arrays to save all values of the losses between real distribution and output of Z-network
        if it == 0:
            metrics = np.array([[lossreale.numpy(),lossrealw.numpy(),lossrealk.numpy()]])
            c_distributions = np.array([tf.nn.softmax(dist()).numpy()[0]])
        else:
            metrics = np.append(metrics, [[lossreale.numpy(),lossrealw.numpy(),lossrealk.numpy()]], axis=0)
            c_distributions = np.append(c_distributions, [tf.nn.softmax(dist()).numpy()[0]], axis=0)

        it = it + 1
    
    # save everything (have to have folder ./data)
    with open('data/conv_distr_50_0_50_' + Divergence + '_' + str(exec) + '.npy', 'wb') as f:
        np.save(f, c_distributions)

    with open('data/conv_metrics_50_0_50_' + Divergence + '_' + str(exec) + '.npy', 'wb') as f:
        np.save(f, metrics)

