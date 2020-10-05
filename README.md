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
 * **[Keras](https://keras.io/) 2.1.0** 
[torch]  
  
## Installation guide
#### **Operation system**  
Ubuntu 16.04 download from https://www.ubuntu.com/download/desktop  
#### **Python and packages**  
Download Anaconda 3-5.2.0 tarball on https://www.anaconda.com/distribution/#download-section  

#Dataset：
reference：CRISPR-Local: a local single-guide RNA (sgRNA) design tool for non-reference plant genomes.
download:http://crispr.hzau.edu.cn/CRISPR-Local/.    


**Note:**  
* The input training and testing files should include sgRNA sequence with length of 23 bp and four "A-N" symbolic corresponding epigenetic features seuqnces with length of 23 as well as label in each gRNA sequence.    

## Demo instructions  
#### **Input (gRNA sequence and four epigenetic features):**               
* #### **Data format:**      
*   **sgRNA sequence:** TGAGAAGTCTATGAGCTTCAAGG (23bp)     


