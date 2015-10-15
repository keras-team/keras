from __future__ import absolute_import
from __future__ import print_function

import tarfile, inspect, os
from six.moves.urllib.request import FancyURLopener

from ..utils.generic_utils import Progbar

class ParanoidURLopener(FancyURLopener):
  def http_error_default(self, url, fp, errcode, errmsg, headers):
    raise Exception('URL fetch failure on {}: {} -- {}'.format(url, errcode, errmsg))

def get_file(fname, origin, untar=False):
    datadir = os.path.expanduser(os.path.join('~', '.keras', 'datasets'))
    if not os.path.exists(datadir):
        os.makedirs(datadir)

    if untar:
        untar_fpath = os.path.join(datadir, fname)
        fpath = untar_fpath + '.tar.gz'
    else:
        fpath = os.path.join(datadir, fname)

    try:
        f = open(fpath)
    except:
        print('Downloading data from',  origin)

        global progbar
        progbar = None

        def dl_progress(count, block_size, total_size):
            global progbar
            if progbar is None:
                progbar = Progbar(total_size)
            else:
                progbar.update(count*block_size)

        ParanoidURLopener().retrieve(origin, fpath, dl_progress)
        progbar = None

    if untar:
        if not os.path.exists(untar_fpath):
            print('Untaring file...')
            tfile = tarfile.open(fpath, 'r:gz')
            tfile.extractall(path=datadir)
            tfile.close()
        return untar_fpath

    return fpath
