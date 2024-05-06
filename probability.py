#!/usr/bin/env python3

from acme.tf.networks import distributions
import tensorflow as tf

def wasserstein_dist(
        p1: distributions.DiscreteValuedDistribution,
        p2: distributions.DiscreteValuedDistribution,
        offset: tf.Tensor=tf.constant(0,tf.float32,shape=[1]),
        scale: tf.Tensor=tf.constant(1.0,tf.float32,shape=[1]),
        exponent=2.0,
        ):
    """
    The wasserstein distance is the integral over the difference
    between the two quantile function of both distributions.
    The quantile functions are the inverse of the cumulative
    distribution functions.
    Since here we consider discrete distributions with arbitrary but
    finite support, the cumulative distribution functions are step
    functions. The inverse of these (the quantile functions) are
    thereby also step functions, and so is the difference between
    them.
    The integral over the difference is then given by the length of
    each step multiplied by its hight.
    We cumpute the x and respective y values of the steps to be able to
    compute the desired integral.
    The x-values of the quantile functions are the y values of the
    cumulative distribution functions and vice versa.

    Expects batch dim in p1, p2 and offset

    """
    dist_vals = tf.expand_dims(p2.values,0)
    dist_vals = tf.repeat(dist_vals,offset.shape[0],0)
    target_vals = dist_vals * tf.expand_dims(scale,1) + tf.expand_dims(offset,1)

    p1_probs = p1.probs
    if p1_probs is None:
        p1_probs = tf.math.softmax(p1.logits)
    p2_probs = p2.probs
    if p2_probs is None:
        p2_probs = tf.math.softmax(p2.logits)
    #The y values of the cumulative distribution functions are the
    #cumulative sum of the probabilities. They start at 0 and end at 1.
    #This cumulative sum doesn't include the 0 at the beginning but does
    #include the 1 end.
    dist_fun = tf.cumsum(p1_probs,1)
    target_dist_fun = tf.cumsum(p2_probs,1)
    #To get all x values of the steps of the quantile
    #function difference, we merge the y values of both cum dist funs
    #and cut of the "1"s at the end of them
    dist_fun_vals = tf.concat(
            [dist_fun[:,:-1],target_dist_fun[:,:-1]]
            ,1)
    #Then we sort all the x values and keep track of the permutation
    #to later identify the correct y values to the x values
    dist_fun_idxs = tf.argsort(dist_fun_vals)
    dist_fun_vals = tf.gather(dist_fun_vals,dist_fun_idxs,batch_dims=-1)
    #finally add a "0" to the beginning since this is not produced
    #in the cumsum and add back the "1"
    dist_fun_vals = tf.concat(
            [
                tf.zeros([dist_fun_vals.shape[0],1]),
                dist_fun_vals,
                tf.ones([dist_fun_vals.shape[0],1]),
            ], 1,)
    #These differences are the lengths of the x-intervalls in our
    #integral
    dist_fun_diffs = dist_fun_vals[:,1:] - dist_fun_vals[:,:-1]
    
    #we produce the y values, which are the differences in the
    #quantile functions by adding up the steps in the step function.
    #The changes given a respective x value from dist_fun_vals are
    #The change of the corresponding quantile function for that x value
    val_diffs_1 = dist_vals[:,1:] - dist_vals[:,:-1]
    val_diffs_2 = target_vals[:,1:] - target_vals[:,:-1]
    #Since we want the difference of the quantile functions, the steps
    #of one of them has to be negated
    vals = tf.concat([val_diffs_1,-val_diffs_2],1)
    #order the y values to correspond to the correct x values.
    vals = tf.gather(vals,dist_fun_idxs,batch_dims=-1)
    #the following initializes the value of the quantile function
    #difference as it is the first summand in the cumsum
    #notice that we also concatinated a number to dist_fun_vals after
    #sorting it.
    #Here we concat the initial value of the quantile difference,
    #to which its stepwise changes are added in the cumsum
    vals = tf.concat(
            [
                tf.expand_dims(dist_vals[:,0] - target_vals[:,0],1),
                vals,
            ], 1,)
    #to this initial value, the changes of the quantile function
    #difference are added thereby producing the desired result
    vals = tf.cumsum(vals,1)
    #transform the y values of the integral the desired way
    moved = tf.pow(tf.abs(vals),exponent)
    #multiply y values by intervall lengths
    distance_moved = dist_fun_diffs * moved
    #integrate by summation
    return tf.reduce_sum(distance_moved,1)

def energy_dist(
        p1: distributions.DiscreteValuedDistribution,
        p2: distributions.DiscreteValuedDistribution,
        offset: tf.Tensor=tf.constant(0,tf.float32,shape=[1]),
        scale: tf.Tensor=tf.constant(1.0,tf.float32,shape=[1]),
        exponent=2.0,
        ):

    dist_vals = tf.expand_dims(p2.values,0)
    dist_vals = tf.repeat(dist_vals,tf.shape(offset)[0],0)
    target_vals = dist_vals * tf.expand_dims(scale,1) + tf.expand_dims(offset,1)
    '''
    dist_vals = p2.values
    target_vals = dist_vals * tf.expand_dims(scale,1) + tf.expand_dims(offset,1)
    '''

    #why is the computation of the probs not handled inside tfp???
    p1_probs = p1.probs
    if p1_probs is None:
        p1_probs = tf.math.softmax(p1.logits)
    p2_probs = p2.probs
    if p2_probs is None:
        p2_probs = tf.math.softmax(p2.logits)

    vals = tf.concat([dist_vals,target_vals],1)
    idxs = tf.argsort(vals)
    vals = tf.gather(vals,idxs,batch_dims=-1)

    probs = tf.concat([p1_probs, -p2_probs],1)
    probs = tf.gather(probs,idxs,batch_dims=-1)
    #the entire sum is 0 because both probs sum to 1
    #therefore we can cut the cumsum one short
    #this makes integration later possible because the val_steps
    #also are 1 shorter
    cum_distr_diffs = tf.cumsum(probs[:,:-1],1)
    if exponent == 2.0:
        trafo_diffs = tf.square(cum_distr_diffs)
    elif exponent == 1.0:
        trafo_diffs = tf.abs(cum_distr_diffs)
    else:
        trafo_diffs = tf.math.pow(tf.abs(cum_distr_diffs),exponent)

    val_steps = vals[:,1:] - vals[:,:-1]
    integrand = val_steps * trafo_diffs
    integral = tf.reduce_sum(integrand,1)
    return integral
    


