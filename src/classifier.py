import pandas as pd
import numpy as np
import tensorflow as tf
import argparse

from os.path import join, dirname
from models import LSTM_Model

class Logger:
    '''
    Object which handles logging learning statistics to file
    during learning. Prints to learning_summary.json in
    specified directory
    '''

    def __init__(self, outdir):
        self.outfile = join(outdir, 'learning_summary.json')

    def log(self, data):
        '''
        Logs dictionary to json file - one object per line
        Args:
            data : dictionary of attribute value pairs
        '''
        data = {str(k): str(v) for k, v in data.iteritems()}
        self._prettyPrint(data)
        with open(self.outfile, 'a') as out:
            out.write(json.dumps(data))
            out.write('\n')

    def log_config(self, opts):
        '''Logs GlobalOpts Object or any object as dictionary'''
        data = {str(k): str(v) for k, v in opts.__dict__.iteritems()}
        self.log(data)

    def _prettyPrint(self, data):
        '''Prints out a dictionary to stout'''
        vals = [str(k) + ':' + str(v) for k, v in data.iteritems()]
        print ' - '.join(vals)

class GlobalOpts(object):
    def __init__(self, name):
        # Directory structure
        self.project_dir = join(dirname(__file__), '..')
        self.classifier_dir = join(self.project_dir, 'classifiers')
        self.checkpoint_dir = join(self.project_dir, 'checkpoints')
        self.glove_path = join(self.project_dir, 'data', 'glove.42B.300d.txt')
        self.archlog_dir = join(self.project_dir, 'log', name)

        # Print thresholds
        SUMM_CHECK = 50
        VAL_CHECK = 200
        CHECKPOINT = 10000

class WordCNNOpts(GlobalOpts):
    def __init__(self, name):
        super(WordCNNOpts, self).__init__(name)
        self.batch_size = 32
        self.window_size = 10
        self.num_filters = 50
        self.sentence_len = 1500

class LSTMOpts(GlobalOpts):
    def __init__(self, name):
        super(LSTMOpts, self).__init__(name)
        # Hyperparameters for model
        self.init_scale = 0.04
        self.learning_rate = 1.0
        self.max_grad_norm = 10
        self.num_layers = 2
        self.num_steps = 50
        self.hidden_size = 1500
        self.max_epoch = 14
        self.max_max_epoch = 55
        self.keep_prob = 0.35
        self.lr_decay = 1 / 1.15
        self.batch_size = 32



if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Text Classification Models for PE-Predictive Project')
    parser.add_argument('--arch', help='Network architecture',
                        type=str, required=True)
    parse.add_arguement('--name', help='Name of directory to place output files in',
                        type=str, required=True)
    args = parser.parse_args()

    opts = LSTMOpts(args.name)
    #data_paths = [join(opts.classifier_dir, 'chapman-data/chapman_df.tsv'),
    #                    join(opts.classifier_dir, 'stanford-data/stanford_df.tsv')]
    data_paths = [join(opts.classifier_dir, 'stanford-data/stanford_df.tsv')]

    sampler = Sampler(data_paths=data_paths, embedding_path=opts.glove_path)
    embedding_np = sampler.get_embeddings()
    model = LSTM_Model(embedding_np, opts)

    # Train model
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for it in range(opts.MAX_ITERS):

            # Step of optimization method
            batchX, batchy = sampler.sample_train()
            train_loss, train_acc = model.step(batchX, batchy, train=True)

            if it % SUMM_CHECK == 0:
                logger.log({'iter': it, 'mode': 'train','dataset': 'train',
                    'loss': train_loss, 'acc': train_acc})
            if it != 0 and it % VAL_CHECK == 0:
                # Calculate validation accuracy
                batchX, batchy = sampler.sample_val()
                val_loss, val_acc = model.step(batchX, batchy, train=False)
                logger.log({'iter': it, 'mode': 'train', 'dataset': 'val',
                            'loss': val_loss, 'acc': val_acc})
            if (it != 0 and it % CHECKPOINT == 0) or \
                    (it + 1) == opts.MAX_ITERS:
                model.save_weights(it)


    # Evaluate performance of model
    with tf.Session() as sess:
        model.restore_weights()
