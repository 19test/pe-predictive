from os.path import join, dirname

class GlobalOpts(object):
    def __init__(self, name):
        # Directory structure
        self.project_dir = join(dirname(__file__), '..', '..','..')
        self.classifier_dir = join(self.project_dir, 'classifiers')
        self.cnn_model_dir = join(self.classifier_dir, 'cnn_model')
        self.checkpoint_dir = join(self.cnn_model_dir, 'checkpoints')
        self.data_dir = join(self.project_dir, 'data')
        self.task1_annot_dir = join(self.data_dir, 'task1_annotations')
        self.task2_annot_dir = join(self.data_dir, 'task2_annotations')
        self.glove_path = join(self.data_dir, 'glove.42B.300d.txt')
        self.archlog_dir = join(self.cnn_model_dir, 'log', name)
        self.partition_dir = join(self.cnn_model_dir, 'partition')
        
        self.report_data_path = join(self.data_dir, 'stanford_pe.tsv')

        # Print thresholds
        self.SUMM_CHECK = 50
        self.VAL_CHECK = 200
        self.CHECKPOINT = 10000
        self.MAX_ITERS = 5000

        self.init_lr = 0.001
        self.decay_factor = 0.1
        self.decay_steps = 1000

        # Common hyperparameters across all models
        self.full_report = False
        self.batch_size = 32
        self.sentence_len = 1500

    def add_args(self, args):
        self.full_report = args.full_report
        # full report - 99.5 percentile - 2026 words
        # impressions - 99.5 percentile -275 words
        self.sentence_len = 2000 if args.full_report else 300
        self.partition = args.partition
        self.error_analysis = args.error_analysis

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

    def get_opts(self, args):
        self.opts.add_args(args)
        return self.opts

    def get_model(self, embedding_np, task_num):
        if task_num == 1:
            from task1_models import LSTM_Model, CNN_Word_Model
        elif task_num == 2:
            from task2_models import LSTM_Model, CNN_Word_Model
        else:
            raise Exception('Invalid Task Number : %d'%task_num)

        if self.arch == 'lstm':
            return LSTM_Model(self.opts, embedding_np)
        elif self.arch == 'cnn_word':
            return CNN_Word_Model(self.opts, embedding_np)
        else:
            raise Exception('Invalid Model Architecture : %s'%self.arch)
