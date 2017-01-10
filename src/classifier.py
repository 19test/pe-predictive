import tensorflow as tf
import pandas as pd
import numpy as np
import argparse
import os
import json

from os.path import join, dirname
from models import LSTM_Model, CNN_Word_Model
from reader import Reader

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
        self.partition_dir = join(self.project_dir, 'partition')
        
        self.data_path = join(self.project_dir, 'data', 'stanford_pe.tsv')

        # Print thresholds
        self.SUMM_CHECK = 50
        self.VAL_CHECK = 200
        self.CHECKPOINT = 10000
        self.MAX_ITERS = 20000

        # Common hyperparameters across all models
        self.full_report = False
        self.batch_size = 32
        self.sentence_len = 1500

class WordCNNOpts(GlobalOpts):
    def __init__(self, name):
        super(WordCNNOpts, self).__init__(name)
        self.window_size = 10
        self.num_filters = 50
        self.keep_prob = 0.5

class LSTMOpts(GlobalOpts):
    def __init__(self, name):
        super(LSTMOpts, self).__init__(name)
        # Hyperparameters for model
        self.init_scale = 0.04
        self.learning_rate = 1.0
        self.max_grad_norm = 10
        self.num_layers = 2
        self.num_steps = 50
        # Should be the same size as word embedding
        self.hidden_size = 300
        self.max_epoch = 14
        self.max_max_epoch = 55
        self.keep_prob = 0.35
        self.lr_decay = 1 / 1.15
        self.batch_size = 32

class ModelFactory(object):
    def __init__(self, arch, name):
        if arch == 'lstm':
            self.opts = LSTMOpts(name)
        elif arch == 'cnn_word':
            self.opts = WordCNNOpts(name)
        else:
            raise Exception('Input architecture not supported : %s' % args.arch)
        self.arch = arch
        self.name = name

    def get_opts(self):
        return self.opts

    def get_model(self, embedding_np):
        if self.arch == 'lstm':
            return LSTM_Model(self.opts, embedding_np)
        elif self.arch == 'cnn_word':
            return CNN_Word_Model(self.opts, embedding_np)
        assert False

if __name__ == '__main__':
    tf.set_random_seed(1)
    np.random.seed(1)

    parser = argparse.ArgumentParser(
        description='Text Classification Models for PE-Predictive Project')
    parser.add_argument('--runtype', help='train or test',
            type=str, required=True)
    parser.add_argument('--arch', help='Network architecture',
            type=str, required=True)
    parser.add_argument('--name',
            help='Name of directory to place output files in',
            type=str, required=True)
    parser.add_argument('--partition',
            help='Way to split data into train/val/test set', required=True)
    parser.add_argument('-full_report', action='store_true',
            help='use full report text - otherwise use impression input')
    args = parser.parse_args()
    factory = ModelFactory(args.arch, args.name)
    opts = factory.get_opts()
    opts.partition = args.partition
    opts.full_report = args.full_report


    if not os.path.exists(opts.archlog_dir):
        os.makedirs(opts.archlog_dir)
    logger = Logger(opts.archlog_dir)

    reader = Reader(opts=opts)
    embedding_np = reader.get_embedding(opts.glove_path)
    model = factory.get_model(embedding_np)

    if args.runtype == 'train':
        # Train model
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for it in range(opts.MAX_ITERS):

                # Step of optimization method
                batchX, batchy = reader.sample_train()
                train_loss, train_acc = model.step(batchX, batchy, train=True)

                if it % opts.SUMM_CHECK == 0:
                    logger.log({'iter': it, 'mode': 'train','dataset': 'train',
                        'loss': train_loss, 'acc': train_acc})
                if it != 0 and it % opts.VAL_CHECK == 0:
                    # Calculate validation accuracy
                    batchX, batchy = reader.sample_val()
                    val_loss, val_acc = model.step(batchX, batchy, train=False)
                    logger.log({'iter': it, 'mode': 'train', 'dataset': 'val',
                                'loss': val_loss, 'acc': val_acc})
                if (it != 0 and it % opts.CHECKPOINT == 0) or \
                        (it + 1) == opts.MAX_ITERS:
                    model.save_weights(it)

    elif args.runtype == 'test':
        # Evaluate performance of model
        with tf.Session() as sess:
            model.restore_weights()
            result = None
            gt = None
            for batchX, batchy in reader.get_test_batches():
                output = model.predict(batchX)
                if result is None:
                    result = output
                    gt = batchy
                else:
                    result = np.vstack((result, output))
                    gt = np.vstack((gt, batchy))
            test_acc = np.mean(result == gt)
            test_prec = np.mean(gt[result==1])
            test_recall = np.mean(result[gt==1])
            print 'Test Set Evaluation'
            print 'Accuracy : %f' % test_acc
            print 'Precision : %f' % test_prec
            print 'Recall : %f' % test_recall

    else:
        raise Exception('Unsupported Runtype : %s' % args.runtype)
