import random

import torch
import numpy as np
from sklearn.metrics import confusion_matrix

## Fix Random Seed
def set_seeds(seed):
    # Python built-in random module
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Torch
    torch.manual_seed(seed)
    # Cuda
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
## Macro-F1 Score
def cal_f1_score(y_pred, y_true):
    """ Precision_Recall_F1score metrics
    y_pred: the predicted score of each class, shape: (Batch_size, num_classes)
    y_true: the ground truth labels, shape: (Batch_size,) for 'multi-class' or (Batch_size, n_classes) for 'multi-label'
    """
    eps=1e-20
    y_pred = torch.argmax(y_pred, dim=1)

    y_pred = y_pred.numpy()
    y_true = y_true.numpy()

    # F1_sci = f1_score(y_true, y_pred, average=None)
    confusion = confusion_matrix(y_true, y_pred)

    f1_list = []
    precision_list = []
    for i in range(len(confusion)):
        TP = confusion[i, i]
        FP = sum(confusion[i, :]) - TP
        FN = sum(confusion[:, i]) - TP

        precision = TP / (TP + FN + eps)
        recall = TP / (TP + FP + eps)
        result_f1 = 2 * precision  * recall / (precision + recall + eps)

        f1_list.append(result_f1)
        precision_list.append(precision)
    
    f1_list = np.array(f1_list)
    Macro_f1 = np.mean(f1_list)

    return Macro_f1        
