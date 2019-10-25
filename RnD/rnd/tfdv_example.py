from __future__ import print_function
import warnings
warnings.filterwarnings('ignore',category=FutureWarning) # to ignore the numpy warnings
import sys, os
import tempfile, urllib, zipfile
import tensorflow as tf
import tensorflow_data_validation as tfdv

tf.logging.set_verbosity(tf.logging.ERROR)
print('TFDV version: {}'.format(tfdv.version.__version__))

# Confirm that we're using Python 3
assert sys.version_info.major is 3, 'Oops, not running Python 3. Use Runtime > Change runtime type'

# Set up some globals for our file paths
BASE_DIR = "/home/neo/tmp_data"
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'chicago_taxi_output')
TRAIN_DATA = os.path.join(DATA_DIR, 'train', 'data.csv')
EVAL_DATA = os.path.join(DATA_DIR, 'eval', 'data.csv')
SERVING_DATA = os.path.join(DATA_DIR, 'serving', 'data.csv')

# Download the zip file from GCP and unzip it
zip, headers = urllib.request.urlretrieve('https://storage.googleapis.com/tfx-colab-datasets/chicago_data.zip')
zipfile.ZipFile(zip).extractall(BASE_DIR)
zipfile.ZipFile(zip).close()

print("Here's what we downloaded:")
os.system('ls -R /home/neo/tmp_data/data')

train_stats = tfdv.generate_statistics_from_csv(data_location=TRAIN_DATA)
tfdv.visualize_statistics(train_stats)
schema = tfdv.infer_schema(statistics=train_stats)
tfdv.display_schema(schema=schema)

