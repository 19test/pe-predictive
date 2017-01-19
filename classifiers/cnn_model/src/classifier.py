import tensorflow as tf
import pandas as pd
import numpy as np
import argparse
import os
from os.path import join

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
                result = np.vstack((result, output))
                gt = np.vstack((gt, batchy))
        result = result[0:test_size,:]
        gt = gt[0:test_size,:]

        assert result.shape[0] == test_size
        assert gt.shape[0] == test_size
        if opts.error_analysis:
            print_error_analysis(reader, result, gt)
        return result, gt

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

def calculate_metrics(predictions, ground_truth, task_num):
    def _get_dimension_metrics(predictions, ground_truth):
        accuracy = np.mean(predictions == ground_truth)
        precision = np.mean(ground_truth[predictions==1])
        recall = np.mean(predictions[ground_truth==1])
        return accuracy, precision, recall
    assert task_num in [1,2]
    if task_num == 1:
        accuracy, precision, recall = \
                _get_dimension_metrics(predictions, ground_truth)
        return {
                'accuracy':accuracy,
                'precision':precision,
                'recall':recall
                }
    else:
        accuracy_pe, precision_pe, recall_pe = \
                _get_dimension_metrics(predictions[:,0], ground_truth[:,0])
        accuracy_burden, _, _ = \
                _get_dimension_metrics(predictions[:,1], ground_truth[:,1])
        return {
                'accuracy_pe':accuracy_pe,
                'precision_pe':precision_pe,
                'recall_pe':recall_pe,
                'accuracy_burden':accuracy_burden,
                }

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
    parser.add_argument('--task_num',
            help='Either task 1 - impressions, or task 2 - full report text',
            type=int, required=True)

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
    model = factory.get_model(embedding_np, task_num=opts.task_num)

    if args.runtype == 'train':
        train(model, reader, logger, opts)

    elif args.runtype == 'test':
        result, gt = test(model, reader, logger, opts)
        metrics = calculate_metrics(result, gt, task_num=opts.task_num)
        if opts.task_num == 2:
            df = pd.DataFrame({
                'gt_pe':gt[:,0],
                'gt_burden':gt[:,1],
                'pred_pe':result[:,0],
                'pred_burden':result[:,1]
                })
            df.to_csv(join(opts.data_dir, 'task2_test.csv'))

        print 'Test Set Evaluation'
        for metric in metrics:
            print '%s : %f' % (metric, metrics[metric])

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
