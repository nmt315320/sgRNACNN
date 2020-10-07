# sgRNACNN
 ## Overview
sgRNACNN is a deep learning-based method for sgRNA on-target activity prediction. 

## Pre-requisite:  
* **Ubuntu 16.04**
* **Anaconda 3-5.2.0**
* **Python packages:**   
  [numpy](https://numpy.org/) 1.16.4  
  [pandas](https://pandas.pydata.org/) 0.23.0  
  [scikit-learn](https://scikit-learn.org/stable/) 0.19.1  
  [scipy](https://www.scipy.org/) 1.1.0  
 **[Keras](https://keras.io/) 2.1.0** 
[torch]  
  
## Installation guide
#### **Operation system**  
Ubuntu 16.04 download from https://www.ubuntu.com/download/desktop  
#### **Python and packages**  
Download Anaconda 3-5.2.0 tarball on https://www.anaconda.com/distribution/#download-section  

#Dataset：
reference：CRISPR-Local: a local single-guide RNA (sgRNA) design tool for non-reference plant genomes.
download:http://crispr.hzau.edu.cn/CRISPR-Local/.    
After downloading the data set, use cd-hit to de-redundant, where the positive and negative thresholds are 0.4 and 0.3 respectively.
We conducted an independent test set verification on the four crops. In the independent test set verification, Randomly select 80% of the data set as the training set and 20% as the test set.

Using：
Including two py files, train.py and test.py;
First run train.py, and then get 5 separate models and an integrated model; then run test.py to get the result.

**Note:**  
* The input training and testing files should include sgRNA sequence with length of 23 bp and four "A-N" symbolic corresponding epigenetic features seuqnces with length of 23 as well as label in each gRNA sequence.    

## Demo instructions  
#### **Input (gRNA sequence and four epigenetic features):**               
* #### **Data format:**      
*   **sgRNA sequence:** TGAGAAGTCTATGAGCTTCAAGG (23bp)    
#output 
==> Loading 
  (c1_1): Conv1d(8, 32, kernel_size=(3,), stride=(1,), padding=(1,))
  (c1_1bn): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (c1_2): Conv1d(32, 32, kernel_size=(3,), stride=(1,), padding=(1,))
  (c1_2bn): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (c1_3): Conv1d(32, 32, kernel_size=(3,), stride=(1,), padding=(1,))
  (c1_3bn): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (p1): MaxPool1d(kernel_size=4, stride=4, padding=0, dilation=1, ceil_mode=False)
  (c2_1): Conv1d(32, 64, kernel_size=(3,), stride=(1,), padding=(1,))
  (c2_1bn): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (c2_2): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(1,))
  (c2_2bn): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (c2_3): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(1,))
  (c2_3bn): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (p2): MaxPool1d(kernel_size=4, stride=4, padding=0, dilation=1, ceil_mode=False)
  (fc): Linear(in_features=64, out_features=256, bias=True)
  (out): Linear(in_features=256, out_features=1, bias=True)
  (criterion): BCELoss()
)

############### EPOCH :  0  ###############
epoch_loss_train_avg:  0.6329017978761009
epoch loss avg:  0.5023078798043608
auc:  0.8373854667513381
acc:  0.7491582491582491
Save model, best_val_loss:  0.5023078798043608

############### EPOCH :  1  ###############
epoch_loss_train_avg:  0.5176053265111228
epoch loss avg:  0.4916402378467598
auc:  0.846015150140615
acc:  0.7626262626262627
Save model, best_val_loss:  0.4916402378467598

############### EPOCH :  2  ###############
nb_train:  2374
epoch_loss_train_avg:  0.4953221738489114
nb_samples:  594
epoch loss avg:  0.4923407738457625
arr_prob:  594
arr_labels:  594
auc:  0.8478295382382292
acc:  0.7525252525252525



