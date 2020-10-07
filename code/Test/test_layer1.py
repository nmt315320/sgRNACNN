# Authors: PhD. Nguyen Hong Quang
# School of Information and Communication Technology
# Hanoi University of Science and Technology
# Email: quangnh@soict.hust.edu.vn
import numpy as np

import torch 
from torch import nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import pickle
import time
import math
import os
import csv
import glob

from sklearn import metrics
from sklearn.model_selection import KFold, StratifiedKFold

from sklearn.metrics import confusion_matrix   #计算混淆矩阵

from sklearn.metrics import matthews_corrcoef #计算MCC

############# HYPER-PARAMETERS ############
FILE_MODEL_TMP = "model_tmp.pkl"

MY_RANDOM_STATE = 5 
torch.manual_seed(MY_RANDOM_STATE)

SAMPLE_LENGTH = 23
AVGPOOL1D_KERNEL_SIZE = 4
CONV1D_KERNEL_SIZE = 3
CONV1D_FEATURE_SIZE_BLOCK1 = 32
CONV1D_FEATURE_SIZE_BLOCK2 = 64
CONV1D_FEATURE_SIZE_BLOCK3 = 128

FULLY_CONNECTED_LAYER_SIZE = 256

MODEL_DIR = '../Train/model_layer1_seed' + str(MY_RANDOM_STATE)
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
###########################################
my_dict = {'A': 0,
        'C': 1, 
        'G': 2,
        'T':3,
        'a':0,
        'c':1,
        'g':2,
        't':3}

# data = one_hot(1,3) ==> [0. 1. 0.]        
def one_hot(index, dimension):
    data = np.zeros((dimension))
    data[index] = 1
    return data

#data = one_hot(1,3)
#print(data)


def load_text_file(file_text):
    with open(file_text) as f:
        lines = f.readlines()
        my_data = [line.strip().upper() for line in lines[1::2]]
        return my_data

class EnhancerDataset(Dataset):
    # X: list of Enhancer sequences (200 characters for each sequence)
    # Y: list label [0, 1]; 0: negative, 1: positive
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __getitem__(self, index):
        label = self.Y[index]
        sample = self.X[index]
        
        values = np.zeros((4, SAMPLE_LENGTH))
        for i in range(SAMPLE_LENGTH):
            char_idx = my_dict[sample[i]]
            values[char_idx, i] = 1 
        
        values_one_mer = self.extract_1_mer(sample)
        #values = np.concatenate((values, values_one_mer), axis=0)
        values_two_mer = self.extract_2_mer(sample)
        #values = np.concatenate((values, values_two_mer), axis=0)
        values_three_mer = self.extract_3_mer(sample)
        #values = np.concatenate((values, values_three_mer), axis=0)
        values = np.concatenate((values, values_one_mer, values_two_mer, 
                        values_three_mer), axis=0)
        
        input = torch.from_numpy(values)
        return input, label
    
    def extract_1_mer(self, sample):
        my_count = {'A': 0.0, 'C': 0.0, 'G': 0.0, 'T': 0.0}        
        values = np.zeros((1, SAMPLE_LENGTH))
        for i in range(SAMPLE_LENGTH):
            my_count[sample[i]] += 1
        
        #for one_mer in my_count:
        #    print("one mer: ", one_mer, " : ", my_count[one_mer])
        
        for i in range(SAMPLE_LENGTH):
            values[0, i] = my_count[sample[i]] / SAMPLE_LENGTH;
        
        #print("values: ", values)    
        return values
    
    def extract_2_mer(self, sample):
        my_count = {'AA': 0.0, 'AC': 0.0, 'AG': 0.0, 'AT': 0.0,
                    'CA': 0.0, 'CC': 0.0, 'CG': 0.0, 'CT': 0.0,
                    'GA': 0.0, 'GC': 0.0, 'GG': 0.0, 'GT': 0.0,
                    'TA': 0.0, 'TC': 0.0, 'TG': 0.0, 'TT': 0.0} 
        values = np.zeros((2, SAMPLE_LENGTH))
        for i in range(SAMPLE_LENGTH - 1):
            two_mer = sample[i:i+2]
            #print("two_mer: ", two_mer)
            my_count[two_mer] += 1
        
        #for two_mer in my_count:
        #    print("two mer: ", two_mer, " : ", my_count[two_mer])
        
        values = np.zeros((2, SAMPLE_LENGTH))
        for i in range(1,SAMPLE_LENGTH-1):
            two_mer_left = sample[i-1:i+1]
            two_mer_right = sample[i:i+2]
            
            values[0, i] = my_count[two_mer_left] / (SAMPLE_LENGTH - 1);
            values[1, i] = my_count[two_mer_right] / (SAMPLE_LENGTH - 1);
        
        #print("values: ", values) 
        return values
    
    def extract_3_mer(self, sample):
        my_count = {}
                                        
        for firchCh in ['A', 'C', 'G', 'T']:
            for secondCh in ['A', 'C', 'G', 'T']:
                for thirdCh in ['A', 'C', 'G', 'T']:
                    three_mer = firchCh + secondCh + thirdCh
                    my_count[three_mer] = 0.0
        for i in range(SAMPLE_LENGTH - 2):
            three_mer = sample[i:i+3]
            #print("two_mer: ", two_mer)
            my_count[three_mer] += 1
                    
        values = np.zeros((1, SAMPLE_LENGTH))
        for i in range(1,SAMPLE_LENGTH-2):
            three_mer = sample[i-1:i+2]
            values[0, i] = my_count[three_mer] / SAMPLE_LENGTH;
                    
        return values
        
    def __len__(self):
        #return 100
        return len(self.X)
        
class EnhancerCnnModel(nn.Module):
    def __init__(self):
        super(EnhancerCnnModel, self).__init__()
        self.c1_1 = nn.Conv1d(8, CONV1D_FEATURE_SIZE_BLOCK1, CONV1D_KERNEL_SIZE, padding=1)
        self.c1_1bn = nn.BatchNorm1d(CONV1D_FEATURE_SIZE_BLOCK1)
        self.c1_2 = nn.Conv1d(CONV1D_FEATURE_SIZE_BLOCK1, CONV1D_FEATURE_SIZE_BLOCK1, 
            CONV1D_KERNEL_SIZE, padding=1)
        self.c1_2bn = nn.BatchNorm1d(CONV1D_FEATURE_SIZE_BLOCK1)
        self.c1_3 = nn.Conv1d(CONV1D_FEATURE_SIZE_BLOCK1, CONV1D_FEATURE_SIZE_BLOCK1, 
            CONV1D_KERNEL_SIZE, padding=1)    
        self.c1_3bn = nn.BatchNorm1d(CONV1D_FEATURE_SIZE_BLOCK1)
        self.p1 = nn.MaxPool1d(AVGPOOL1D_KERNEL_SIZE)
        
        self.c2_1 = nn.Conv1d(CONV1D_FEATURE_SIZE_BLOCK1, 
            CONV1D_FEATURE_SIZE_BLOCK2, 
            CONV1D_KERNEL_SIZE, padding=1)
        self.c2_1bn = nn.BatchNorm1d(CONV1D_FEATURE_SIZE_BLOCK2)
        self.c2_2 = nn.Conv1d(CONV1D_FEATURE_SIZE_BLOCK2, CONV1D_FEATURE_SIZE_BLOCK2, 
            CONV1D_KERNEL_SIZE, padding=1)
        self.c2_2bn = nn.BatchNorm1d(CONV1D_FEATURE_SIZE_BLOCK2)
        self.c2_3 = nn.Conv1d(CONV1D_FEATURE_SIZE_BLOCK2, CONV1D_FEATURE_SIZE_BLOCK2, 
            CONV1D_KERNEL_SIZE, padding=1)
        self.c2_3bn = nn.BatchNorm1d(CONV1D_FEATURE_SIZE_BLOCK2)
        self.p2 = nn.MaxPool1d(AVGPOOL1D_KERNEL_SIZE)
        
        self.fc = nn.Linear(64, FULLY_CONNECTED_LAYER_SIZE)
        self.out = nn.Linear(FULLY_CONNECTED_LAYER_SIZE, 1)
        
        self.criterion = nn.BCELoss()        
     
    def forward(self, inputs):
        batch_size = inputs.size(0)
        # Turn (batch_size x seq_len) into (batch_size x input_size x seq_len) for CNN
        #inputs = inputs.transpose(1,2)
        #print("inputs size: ", inputs.size())        
        output = F.relu(self.c1_1bn(self.c1_1(inputs)))
        output = F.relu(self.c1_2bn(self.c1_2(output)))
        output = F.relu(self.c1_3bn(self.c1_3(output)))
        output = self.p1(output)
        #print("After p1: ", output.shape) 
        
        output = F.relu(self.c2_1bn(self.c2_1(output)))
        output = F.relu(self.c2_2bn(self.c2_2(output)))
        output = F.relu(self.c2_3bn(self.c2_3(output)))
        output = self.p2(output)
        #print("After p2: ", output.shape)
        
        output = output.view(batch_size, -1)
        #print("Reshape : ", output.shape)
        
        output = F.relu(self.fc(output))
        #print("After FC layer: ", output.shape)  
        
        output = torch.sigmoid(self.out(output))
        #print("Final output (After sigmoid): ", output.shape)
        #print("Final output: ", output)
        
        return output 
    
def evaluate(file_model, loader):
    #model.eval()
    model = EnhancerCnnModel()
    #print("CNN Model: ", model)
    if torch.cuda.is_available(): model.cuda()
    
    model.load_state_dict(torch.load(file_model))
    model.eval()    
    
    epoch_loss = 0.0
    nb_samples = 0
    
    arr_labels = []
    arr_labels_hyp = []
    arr_prob = []
    
    for i, data in enumerate(loader, 0):
        # get the inputs
        inputs, labels = data
        #print("labels: ", labels)
        
        inputs_length = inputs.size()[0]
        nb_samples += inputs_length
        
        arr_labels += labels.squeeze(1).data.cpu().numpy().tolist()

        inputs = inputs.float()
        labels = labels.float()
        
        if torch.cuda.is_available():
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
        else:
            inputs = Variable(inputs)
            labels = Variable(labels)
        
        outputs = model(inputs)
        loss = model.criterion(outputs, labels)
        
        epoch_loss = epoch_loss + loss.item() * inputs_length
        
        arr_prob += outputs.squeeze(1).data.cpu().numpy().tolist()
    
    print("nb_samples: ", nb_samples)
    epoch_loss_avg = epoch_loss / nb_samples
    print("epoch loss avg: ", epoch_loss_avg)
        
    print("arr_prob: ", len(arr_prob))
    print("arr_labels: ", len(arr_labels))
    
    auc = metrics.roc_auc_score(arr_labels, arr_prob)
    print("auc: ", auc)
    
    arr_labels_hyp = [int(prob > 0.5) for prob in arr_prob]
    #print("arr_prob: ", arr_prob)
    #print("arr_labels_hyp: ", arr_labels_hyp)
    arr_labels = [int(label) for label in arr_labels]
    
    acc, confusion_matrix, sensitivity, specificity, mcc = calculate_confusion_matrix(arr_labels, arr_labels_hyp)
    result = {'epoch_loss_avg': epoch_loss_avg, 
                'acc' : acc, 
                'confusion_matrix' : confusion_matrix,
                'sensitivity' : sensitivity,
                'specificity' : specificity,
                'mcc' : mcc,
                'auc' : auc,
                'arr_prob': arr_prob,
                'arr_labels': arr_labels,
                'arr_labels_hyp':arr_labels_hyp
                 }
    print("acc: ", acc)
    print("mcc: ", mcc)
    return result

def calculate_confusion_matrix(arr_labels, arr_labels_hyp):
    corrects = 0
    confusion_matrix = np.zeros((2, 2))

    for i in range(len(arr_labels)):
        confusion_matrix[arr_labels_hyp[i]][arr_labels[i]] += 1

        if arr_labels[i] == arr_labels_hyp[i]:
            corrects = corrects + 1

    acc = corrects * 1.0 / len(arr_labels)
    specificity = confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[1][0])
    sensitivity = confusion_matrix[1][1] / (confusion_matrix[0][1] + confusion_matrix[1][1])
    tp = confusion_matrix[1][1]
    tn = confusion_matrix[0][0]
    fp = confusion_matrix[1][0]
    fn = confusion_matrix[0][1]
    mcc = (tp * tn - fp * fn ) / math.sqrt((tp + fp)*(tp + fn)*(tn + fp)*(tn + fn))
    #print("mcc: ", mcc)
    return acc, confusion_matrix, sensitivity, specificity, mcc
        
# dataset = {"data_test" : testset["data"], "label_test" : testset["label"]}
def check_dataset(dataset):
    print("\n==> Checking dataset")    
    
    # Check data_test
    data_test = dataset["data_test"]
    nb_error_samples = 0
    for sample in data_test:
        if len(sample) != 200:
            nb_error_samples += 1
    if nb_error_samples > 0:
        print("data_teset error: ", nb_error_samples)
    else: print("data_test : OK!")
    
# File test: test_strong_enhancer.txt, test_weak_enhancer.txt
#>Chr11_6627824_6628024
#ATGCTGCCAGAAGGAAAAGGGGTGGAATTAATGAAACTGGAAGGTTGTGGTGCTGGTTTGAGGAG
#TAAAGTATGGGGGCCAAAGTTGGCTATATGCTGGATATGAAGAGGGGGTTAATTCCTTGCAGGTC
#TTCTTGAGATAGAAGTCCAGGCCCTGAGGTGGCAGGCAGCCTGATAGTGAACAGAACCCTTGTGC
#CCATA

def load_text_file(file_text):
    with open(file_text) as f:
        lines = f.readlines()
        my_data = [line.strip().upper() for line in lines[1::2]]
        return my_data


def prepare_test_data():
    print("\n ==> Loading test set")
    data_enhancer = load_text_file('test_strong_enhancer.txt')
    print("data_enhancer: ", len(data_enhancer))
    
    data_non_enhancer = load_text_file('test_non_enhancer.txt')
    print("data_non_enhancer: ", len(data_non_enhancer))
    
    label_enhancer = np.ones((len(data_enhancer),1))
    label_non_enhancer = np.zeros((len(data_non_enhancer), 1))
    
    data = np.concatenate((data_enhancer, data_non_enhancer))
    label = np.concatenate((label_enhancer, label_non_enhancer))
    
    testset = {"data" : data, "label" : label}
    
    return testset

def testing():
    test_dataset = EnhancerDataset(testset["data"], testset["label"])
    test_loader = DataLoader(dataset=test_dataset, batch_size=32,                              
                              shuffle=False, num_workers=4)
          
    with open(testresult_fn, mode='w') as outfile:
        outfile = csv.writer(outfile, delimiter=',')
        outfile.writerow(['model_fn', 'Accuracy score', 'AUC score', 'Sensitivity', 'Specificity'])
        
    list_model_fn = sorted(glob.glob(MODEL_DIR+"/enhancer_*.pkl"))
    #print(list_model_fn)
    y_prob_mtx = []
    
    for model_fn in list_model_fn:
        print(model_fn)
        result = evaluate(model_fn, test_loader)
        #print(result['arr_prob'])
        y_prob_mtx.append(result['arr_prob'])
        #break
        
        with open(testresult_fn, mode='a') as outfile:
            outfile = csv.writer(outfile, delimiter=',')
            outfile.writerow([model_fn, result['acc'], result['auc'], 
                result['sensitivity'], result['specificity']])
    
    
    y_prob_mtx = np.array(y_prob_mtx)
    print("y_prob_mtx: ", y_prob_mtx.shape)
    #print("y_prob_mtx: ", y_prob_mtx)
    
    y_prob_ensemble = [np.mean(y_prob_mtx[:,col]) for col in range(np.size(y_prob_mtx, 1))] 
    y_pred_ensemble = [np.float(each > 0.5) for each in y_prob_ensemble]
    
    y_true = testset["label"]
    auc_score_ensemble = metrics.roc_auc_score(y_true, y_prob_ensemble)
    accuracy_score_ensemble = metrics.accuracy_score(y_true, y_pred_ensemble)
    
    cm = metrics.confusion_matrix(y_true, y_pred_ensemble)
    specificity_ensemble = cm[0,0]/(cm[0,0] + cm[0,1])
    sensitivity_ensemble = cm[1,1]/(cm[1,1] + cm[1,0])

    print("Accuracy score (Testing Set) = ", accuracy_score_ensemble)
    print("ROC AUC score  (Testing Set) = ", auc_score_ensemble)
    print("Sensitivity    (Testing Set) = ", sensitivity_ensemble)
    print("Specificity    (Testing Set) = ", specificity_ensemble)
    
    with open(testresult_fn, mode='a') as outfile:
        outfile = csv.writer(outfile, delimiter=',')
        outfile.writerow(["ensemble", accuracy_score_ensemble, auc_score_ensemble, sensitivity_ensemble, specificity_ensemble])
   
        
##################################
if __name__== "__main__":    
    testresult_fn = MODEL_DIR + "/test_result.csv"
    
    testset = prepare_test_data()    
    
    testing()
