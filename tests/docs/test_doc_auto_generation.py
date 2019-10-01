import os
from docs import autogen
import pytest
from keras import backend as K


if K.backend() != 'tensorflow':
    pytestmark = pytest.mark.skip


def test_docs_in_custom_destination_dir(tmpdir):
    autogen.generate(tmpdir)
    assert os.path.isdir(os.path.join(tmpdir, 'layers'))
    assert os.path.isdir(os.path.join(tmpdir, 'models'))
    assert os.path.isdir(os.path.join(tmpdir, 'examples'))
    assert os.listdir(os.path.join(tmpdir, 'examples'))


if __name__ == '__main__':
    pytest.main([__file__])
