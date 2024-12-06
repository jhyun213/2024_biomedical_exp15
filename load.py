import wfdb
import numpy as np

def load_wfdb(path, patient, remove_nonaami=True):
    types_n = ['N', 'L', 'R', 'e', 'j']
    types_s = ['A', 'a', 'J', 'S']
    types_v = ['V', 'E']
    types_f = ['F']
    types_unknown = ['/', 'f', 'Q']
    types_aami = types_n + types_s + types_v + types_f + types_unknown

    record_path = path+'/'+str(patient)
    # record = wfdb.rdrecord(record_path, smooth_frames=False)
    record = wfdb.rdrecord(record_path)
    sig =record.p_signal[:, record.sig_name.index('MLII')]
    annotations = wfdb.rdann(record_path, 'atr')
    r_peaks = annotations.sample
    types = annotations.symbol
    
    new_annotations = []
    for r_peak, type in zip(r_peaks, types):
        if type not in types_aami and remove_nonaami:
                continue
        if type in types_n:
            type = 0
        else:
            type = 1
        new_annotations.append({'point':r_peak, 'type':type})
    return sig, new_annotations 
    

def get_patients(dataset):
    DS1 = [100, 105, 108, 109, 111, 114, 115, 116, 117, 119, 121, 124, 200, 203, 205, 207, 208, 209, 213, 221, 223, 228]
    DS2 = [101, 103, 106, 112, 113, 118, 122, 123, 201, 202, 210, 212, 213, 214, 219, 220, 222, 230, 231, 232, 233, 234]
    if dataset == 'ds1':
        patients = DS1
    elif dataset == 'ds2':
        patients = DS2
    elif dataset == 'all':
        patients = DS1+DS2
    else:
        patients = [210]
    return patients


class DatasetWFDB():
    def __init__(
        self, 
        path='../physionet.org/files/mitdb/1.0.0', 
        window_range=(150, 150), 
        train_test_ratio=0.8
    ):
        self.patients = get_patients(210)
        heartbeats = []
        labels = []
        for idx, patient in enumerate(self.patients):
            sig, annotations = load_wfdb(path, patient)
            for ann in annotations:
                r_peak = ann['point']
                window = (r_peak-window_range[0], r_peak+window_range[1])
                if window[0] < 0 or window[1] > len(sig):
                    continue
                heartbeats.append(sig[window[0]:window[1]])
                labels.append(ann['type'])
        X = np.stack(heartbeats)
        Y = np.stack(labels)
        ## split train, test set
        np.random.seed(0)
        idx = np.arange(len(labels))
        np.random.shuffle(idx)
        idx_train = idx[:int(train_test_ratio*len(labels))]
        idx_test = idx[int(train_test_ratio*len(labels)):]
        self.X_train, self.Y_train = X[idx_train], Y[idx_train]
        self.X_test, self.Y_test = X[idx_test], Y[idx_test]
