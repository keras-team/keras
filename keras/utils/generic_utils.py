from __future__ import absolute_import
import numpy as np
import time
import sys
import six


def get_from_module(identifier, module_params, module_name, instantiate=False, kwargs=None):
    if isinstance(identifier, six.string_types):
        res = module_params.get(identifier)
        if not res:
            raise Exception('Invalid ' + str(module_name) + ': ' + str(identifier))
        if instantiate and not kwargs:
            return res()
        elif instantiate and kwargs:
            return res(**kwargs)
        else:
            return res
    return identifier


def make_tuple(*args):
    return args


def printv(v, prefix=''):
    if type(v) == dict:
        if 'name' in v:
            print(prefix + '#' + v['name'])
            del v['name']
        prefix += '...'
        for nk, nv in v.items():
            if type(nv) in [dict, list]:
                print(prefix + nk + ':')
                printv(nv, prefix)
            else:
                print(prefix + nk + ':' + str(nv))
    elif type(v) == list:
        prefix += '...'
        for i, nv in enumerate(v):
            print(prefix + '#' + str(i))
            printv(nv, prefix)
    else:
        prefix += '...'
        print(prefix + str(v))


class Progbar(object):
    def __init__(self, target, width=30, verbose=1):
        '''
            @param target: total number of steps expected
        '''
        self.width = width
        self.target = target
        self.sum_values = {}
        self.unique_values = []
        self.start = time.time()
        self.total_width = 0
        self.seen_so_far = 0
        self.output_values = {'prog': 0.0, 'eta': 0}
        self.verbose = verbose

    def update(self, current, values=[]):
        '''
            @param current: index of current step
            @param values: list of tuples (name, value_for_last_step).
            The progress bar will display averages for these values.
        '''
        for k, v in values:
            if k not in self.sum_values:
                self.sum_values[k] = [v * (current - self.seen_so_far), current - self.seen_so_far]
            else:
                self.sum_values[k][0] += v * (current - self.seen_so_far)
                self.sum_values[k][1] += (current - self.seen_so_far)
            # update value for key
            self.output_values[k] = self.sum_values[k][0] / max(1, self.sum_values[k][1])

        self.seen_so_far = current

        # update progress
        self.output_values['prog'] = float(self.seen_so_far)/self.target

        # update eta
        now = time.time()
        if self.seen_so_far:
            time_per_unit = (now - self.start) / self.seen_so_far
        else:
            time_per_unit = 0
        self.output_values['eta'] = time_per_unit*(self.target - self.seen_so_far)
        self.write_output()

    def write_output(self):
        now = time.time()
        if self.verbose == 1:
            prev_total_width = self.total_width
            sys.stdout.write("\b" * prev_total_width)
            sys.stdout.write("\r")

            # print progress bar
            if 'prog' in self.output_values:
                numdigits = int(np.floor(np.log10(self.target))) + 1
                barstr = '%%%dd/%%%dd [' % (numdigits, numdigits)
                bar = barstr % (self.seen_so_far, self.target)
                prog_width = int(self.width*self.output_values['prog'])
                if prog_width > 0:
                    bar += ('='*(prog_width-1))
                    if self.seen_so_far < self.target:
                        bar += '>'
                    else:
                        bar += '='
                bar += ('.'*(self.width-prog_width))
                bar += ']'
                sys.stdout.write(bar)
                self.total_width = len(bar)

            info = ''
            # print eta
            if 'eta' in self.output_values:
                if self.seen_so_far < self.target:
                    info += ' - ETA: %ds' % self.output_values['eta']
                else:
                    info += ' - %ds' % (now - self.start)

            # print all other values
            for k, v in filter(lambda (x, _): x != 'prog' and x != 'eta', self.output_values.items()):
                info += ' - %s: %.4f' % (k, v)
                
            self.total_width += len(info)
            if prev_total_width > self.total_width:
                info += ((prev_total_width-self.total_width) * " ")

            sys.stdout.write(info)
            sys.stdout.flush()

            if self.seen_so_far >= self.target:
                sys.stdout.write("\n")

        if self.verbose == 2:
            if self.seen_so_far >= self.target:
                info = '%ds' % (now - self.start)
                for k in self.sum_values.iterkeys():
                    info += ' - %s: %.4f' % (k, self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                sys.stdout.write(info + "\n")

    def add(self, n, values=[]):
        self.update(self.seen_so_far+n, values)

    def replace_value(self, k, v):
        self.output_values[k] = v
        self.write_output()
