import json
from os.path import join

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
