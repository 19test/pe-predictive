import tensorflow as tf
import pandas as pd
import numpy as np
import argparse
import os

from reader import Reader
from ModelFactory import ModelFactory
from Logger import Logger


def print_error_analysis(reader, pred, gt):
    assert len(pred) == len(gt), [len(pred), len(gt)]
    assert len(reader.testX) == len(pred), [len(reader.testX), len(pred)]
    total = np.sum(pred != gt)
    counter = 0
    for ind in range(len(pred)):
        if pred[ind] != gt[ind]:
            text = reader.testX[ind]
            counter += 1
            print '----------Example %d / %d ------------' % (counter, total)
            print 'Label : %s - Pred : %s' % (gt[ind], pred[ind])
            print ' '.join([reader.id_to_word[word_id] for word_id in text])

def train(model, reader, logger, opts):
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

def test(model, reader, logger, opts):
    # Evaluate performance of model
    with tf.Session() as sess:
        model.restore_weights()
        result = None
        gt = None
        test_batch_iter = reader.get_test_batches()
        test_size = test_batch_iter.get_num_examples()
        for batchX, batchy in test_batch_iter:
            output = model.predict(batchX)
            if result is None:
                result = output
                gt = batchy
            else:
                result = np.append(result, output)
                gt = np.append(gt, batchy)
        result = result[0:test_size]
        gt = gt[0:test_size]

        assert len(result) == test_size
        assert len(gt) == test_size
        test_acc = np.mean(result == gt)
        test_prec = np.mean(gt[result==1])
        test_recall = np.mean(result[gt==1])
        if opts.error_analysis:
            print_error_analysis(reader, result, gt)
        return test_acc, test_prec, test_recall

def predict(input_batches, model, opts):
    model.restore_weights()

    total_size = input_batches.get_num_examples() 

    result = None
    for batchX, _ in input_batches:
        output = model.predict(batchX)
        if result is None:
            result = output
        else:
            result = np.vstack((result, output))
    result = result[0:total_size,:]
    assert result.shape[0] == total_size
    return result

if __name__ == '__main__':
    tf.set_random_seed(1)
    np.random.seed(1)

    # Required specification of model to be used
    parser = argparse.ArgumentParser(
        description='Text Classification Models for PE-Predictive Project')
    parser.add_argument('--runtype', help='train | test | predict',
            type=str, required=True)
    parser.add_argument('--arch', help='Network architecture',
            type=str, required=True)
    parser.add_argument('--name',
            help='Name of directory to place output files in',
            type=str, required=True)

    # used for train and test runtype
    parser.add_argument('--partition',
            help='Way to split data into train/val/test set')
    
    # used for predict runtype
    parser.add_argument('--input_path',
            help='input csv file with reports to process', type=str)
    parser.add_argument('--output_path',
            help='Path of output csv with additional pred column from input',
            type=str)

    # Additional flags
    parser.add_argument('-full_report', action='store_true',
            help='use full report text - otherwise use impression input')
    parser.add_argument('-error_analysis', action='store_true',
            help='Print text of examples which were predicted incorrectly')
    args = parser.parse_args()
    factory = ModelFactory(args.arch, args.name)
    opts = factory.get_opts(args)


    if not os.path.exists(opts.archlog_dir):
        os.makedirs(opts.archlog_dir)
    logger = Logger(opts.archlog_dir)

    reader = Reader(opts=opts)
    embedding_np = reader.get_embedding(opts.glove_path)
    model = factory.get_model(embedding_np, task_num=1)

    if args.runtype == 'train':
        train(model, reader, logger, opts)
    elif args.runtype == 'test':
        test_acc, test_prec, test_recall = test(model, reader, logger, opts)
        print 'Test Set Evaluation'
        print 'Accuracy : %f' % test_acc
        print 'Precision : %f' % test_prec
        print 'Recall : %f' % test_recall
    elif args.runtype == 'predict':
        sep = '\t' if args.input_path.split('.')[-1]=='tsv' else ','
        input_reports = pd.read_csv(args.input_path, sep=sep)

        labelX_name = 'rad_report' if opts.full_report else 'report_text'
        inputX = input_reports[labelX_name]
        raw_batch_iter = reader.get_raw_batches(inputX)
        predictions = predict(raw_batch_iter, model, opts)
        for i in range(predictions.shape[1]):
            input_reports['pred_%d'%i] = predictions[:,i]
        input_reports.to_csv(args.output_path)
    else:
        raise Exception('Unsupported Runtype : %s' % args.runtype)
