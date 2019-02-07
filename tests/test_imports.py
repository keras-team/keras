import os
from glob import glob
import pytest

#Returns dictionary containing arrays of realtive and absolute imports
def parser(path):
    
    ignore = ['__future__', 'collections', 'random', 'six', 'cPickle', 'scipy', 'hashlib', 'io', 
              'contextlib', 'unittest', 'types', 'h5py', 'inspect', 'tarfile', 'yaml', 'copy', 'marshal',
              'requests', 'functools', 'gzip', 're', 'Queue', 'queue', 'os', 'pickle', 'importlib', 'mock', 'threading',
              'codecs', 'tempfile', 'time', 'binascii', 'pydot', 'zipfile', 'json', 'shutil', 'abc', 'sys', 'csv', 'cntk',
              'warnings', 'numpy', 'skimage', 'multiprocessing', 'distutils', 'tensorflow', 'theano', 'keras_applications', "keras_preprocessing"]
    
    ret = {}
    ret['file'] = path
    ret['absolute'] = []
    ret['relative'] = []
    
    comment = False
    
    with open(path) as p:
        imports = p.read().split('\n')
        
    
    for line in imports:
        line = line.strip()
        #Ingore lines withing multi line comments
        if '\"\"\"' in line:
            comment = not comment
        #Empty line or comment line
        if line == "" or comment == True or '#' in line:
            continue
        else:
            split_line = line.split()
            #Import
            if split_line[0] == 'import':
                module_split = split_line[1].split('.')
                #Check if module is an ignored library
                if module_split[0] in ignore:
                    continue
                else:
                    ret['absolute'].append(line) 
                
            #ImportFrom
            elif split_line[0] == 'from' and len(split_line) > 3:
                #Check if module is an ignored library or line doesnt contain import
                if split_line[1] in ignore or split_line[2] != 'import':
                    continue
                #Check if import is absolute or relative
                elif split_line[1].startswith('.'):
                    ret['relative'].append(line)
                else:
                    module_split = split_line[1].split('.')
                    if module_split[0] in ignore:
                        continue
                    else:
                        ret['absolute'].append(line)
            else:
                continue
        
    return ret


def checkImports(dirPath):
    full = []
    result = [y for x in os.walk(dirPath) for y in glob(os.path.join(x[0], '*.py'))]
    for file in result:
        full.append(parser(file))
    
    return full


def test_imports():
    keras = checkImports("../keras")
    test = checkImports(".")
    
    for imp in keras:
        if len(imp["absolute"]) != 0:
            raise ValueError("Invalid import: " + imp["asbolute"][0] + " in " + imp["file"])
            
    for imp in test:
        if len(imp["relative"]) != 0:
            raise ValueError("Invalid import: " + imp["asbolute"][0] + " in " + imp["file"])
            


