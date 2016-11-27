'''
Creates a LSTM classifier which samples phrases of size N and uses
disease_PEfinder as ground truth
'''
import pandas as pd
import numpy as np

from os.path import join, dirname
from tensorflow.models.rnn.rnn_cell import BasicLSTMCell, MultiRNNCell


### LSTM LAYERS

def lstm(input_data, hidden_size, num_layers, name):
    # input_data : (batch_size, sentence_len, vec_len)
    batch_size, sentence_len, embedding_len = input_data.get_shape()
    input_data = tf.split(1, sentence_len, x_in)
    with tf.variable_scope(name) as scope:
        multi_lstm = MultiRNNCell([BasicLSTMCell(hidden_size)] * num_layers)
        state = multi_lstm.zero_state(batch_size, tf.float32)
        for t in range(sentence_len):
            output, state = multi_lstm(input_data[t], state)
            scope.reuse_variables()
    return output

#TODO : apply gradient clipping
def _train_op_init(total_loss, global_step):
    INITIAL_LEARNING_RATE = 0.001
    LEARNING_RATE_DECAY_FACTOR = 0.1
    DECAY_STEPS = 5000

    self.lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                    global_step,
                                    DECAY_STEPS,
                                    LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)
    opt = tf.train.AdamOptimizer(self.lr)
    grads = opt.compute_gradients(total_loss)
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
    train_ops = [apply_gradient_op]
    train_ops += tf.get_collection('update_ops')
    return train_ops

# Compute softmax cross entropy loss
def softmax_loss(logits, labels):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits,
            labels, name='cross_entropy')
    cross_entropy_mean = tf.reduce_mean(
        cross_entropy, name='mean_cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def dense(input_data, N, H, name):
    """NN fully connected layer."""
    with tf.variable_scope(name):  
        W = tf.get_variable("W", [N, H],
                initializer=tf.contrib.layers.xavier_initializer())   
        b = tf.get_variable("b", [H], initializer=tf.constant_initializer(0))
        return tf.matmul(input_data, W, name="matmul") + b

def dense_relu(input_data, output_dim, name="dense_relu"):
    """NN dense relu layer"""
    batch_size, in_dim = input_data.get_shape()
    with tf.variable_scope(name):
        affine = dense(input_data, in_dim, H, "dense")
        return tf.nn.relu(affine, "relu")

if __name__ == '__main__':
    # Directory structure
    project_dir = join(dirname(__file__), '..')
    classifier_dir = join(project_dir, 'classifiers')
    checkpoint_dir = join(project_dir, 'checkpoints')

    # import chapman and stanford data
    sampler = Sampler()
    sampler.add_data(join(classifier_dir, 'chapman-data/chapman_df.tsv'))
    sampler.add_data(join(classifier_dir, 'stanford-data/stanford_df.tsv'))
    sampler.add_embeddings(join(project_dir, 'data', 'glove.42B.300d.txt')
    sampler.split_train()
    sampler.initialize()

    batch_size = 32
    sentence_len = 50
    embedding_len = sampler.get_embedding_len()

    # create lstm model for learning
    x_in = tf.Placeholder(tf.float32, shape=(batch_size,
        sentence_len, embedding_len))
    y_in = tf.Placeholder(tf.int64, shape=(batch_size))
    global_step = tf.Variable(0, trainable=False)

    logits = lstm(x_in, hiddens_size=200, num_layers=1)
    logits = dense_relu(logits, output_dim=2)
    loss = softmax_loss(logits, gt_labels)
    acc = tf.reduce_mean(tf.equal(tf.argmax(logits, 1), y_in))
    pred = tf.nn.softmax(logits)
    train_op = _train_op_init(loss, global_step=global_step)

    # Train model
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for it in range(opts.MAX_ITERS):

            # Step of optimization method
            batchX, batchy = sampler.sample_train()
            output = sess.run([loss, acc, train_op],
                feed_dict={x_in : batchX, y_in : batchy})
            train_loss, train_acc = (output[0], output[1])

            if it % opts.SUMM_CHECK == 0:
                logger.log({'iter': it, 'mode': 'train','dataset': 'train',
                    'loss': train_loss, 'acc': train_acc})
            if it != 0 and it % opts.VAL_CHECK == 0:
                # Calculate validation accuracy
                batchX, batchy = sampler.sample_val()
                val_loss, val_acc = sess.run([loss, acc],
                    feed_dict={x_in : batchX, y_in : batchy})
                logger.log({'iter': it, 'mode': 'train', 'dataset': 'val',
                            'loss': val_loss, 'acc': val_acc})
            if (it != 0 and it % opts.CHECKPOINT == 0) or \
                    (it + 1) == opts.MAX_ITERS:
                checkpoint_path = join(checkpoint_dir, 'checkpoint.ckpt')
                saver.save(sess, checkpoint_path, global_step=it)


    # Evaluate performance of model
    with tf.Session() as sess:
        all_ckpts = tf.train.get_checkpoint_state(checkpoint_dir)
        saver.restore(sess, all_ckpts.model_checkpoint_path)
        raise Exception('Not yet implemented')
