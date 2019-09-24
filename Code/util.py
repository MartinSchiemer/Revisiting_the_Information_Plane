"""
Author: Martin Schiemer
provides saving and loading functionality
"""

import os
import sys
import errno

try:
    import cPickle as pickle
except:
    import pickle
import gzip

# saving to zip files taken from: http://code.activestate.com/recipes/189972-zip-and-pickle/
# added path creation
def save(object_to_save, filename, protocol = -1):
    """Save an object to a compressed disk file.
       Works well with huge objects.
    """
    if not os.path.exists('Results/Dics/'):
        try:
            os.makedirs('Results/Dics/')
        except OSError as error:
            if error.errno != errno.EEXIST:
                raise
    file = gzip.GzipFile('Results/Dics/'+ filename, 'wb')
    pickle.dump(object_to_save, file, protocol)
    file.close()

def load(filename):
    """Loads a compressed object from disk
    """
    file = gzip.GzipFile('Results/Dics/'+filename, 'rb')
    object_to_load = pickle.load(file)
    file.close()

    return object_to_load

