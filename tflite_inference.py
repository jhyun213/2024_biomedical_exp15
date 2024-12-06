import sys
import argparse
import numpy as np
from time import time
import tflite_runtime.interpreter as tflite
from load import DatasetWFDB

parser = argparse.ArgumentParser()
parser.add_argument(
    '--model',
    required=True,
    help='tflite model path'
)
args = parser.parse_args()

# tflite_path = 'model.tflite'
tflite_path = args.model
interpreter = tflite.Interpreter(tflite_path)
interpreter.allocate_tensors()  # Needed before execution!

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]
in_tensor_idx = input_details['index']
out_tensor_idx = output_details['index']

# dataset_path = '../bnn_ISOCC/mitbih_database/'
dset = DatasetWFDB()
X_val, Y_val = dset.X_test, dset.Y_test
if input_details['dtype'] == np.uint8:
    ## integer quantization
    input_scale, input_zero_point = input_details['quantization']
    for i, x in enumerate(X_val):
        X_val[i] = (x / input_scale + input_zero_point).astype(input_details['dtype'])

t_start = time()
n_total = 0
n_correct = 0
for x, y in zip(X_val, Y_val):
    interpreter.tensor(in_tensor_idx)()[0, :, 0] = x
    interpreter.invoke()
    prob = interpreter.tensor(out_tensor_idx)()[0][0]
    pred = int(prob > 0.5)
    if y == pred:
        n_correct += 1
    n_total += 1
    sys.stdout.write('\r{}/{}'.format(n_total, len(Y_val)))
t_finish = time()
print('\n', n_total, n_correct)
print(n_correct/n_total)
print('time: ', t_finish - t_start)