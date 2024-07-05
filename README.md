# Code for Biological Factor Regulatory Neural Networks for Pyroptosis Therapy in Triple-Negative Breast Cancer

The current repository contains the source code for generating the ranking of drug combinations.


## Installation

The code has been tested with Python 3.9 on macOS 14.5 and includes libraries such as torch, torch_geometric, numpy, networkx, scikit-learn, among others.

pip install -r requirements.txt 

It typically takes a few minutes.

## Running

cd code/

python main.py --score 0.6 --drug1 XXX --drug2 XXX

You need to provide the edge threshold for the protein-protein interaction network, as well as the names of the two drugs you'd like to analyze.

For one drug combination, it takes a few seconds to converage and obtain the result.

Please note that we use torch without cuda to obtain the results. Meanwhile, the code is also compatible with the CUDA version. Users who wish to utilize CUDA should adjust the optimizer and training epochs to ensure the results converge properly.



## Tutorial on New Drug Combinations

To utilize the code for new drug combinations, it is essential to prepare the following files beforehand:

1. drug_target.txt: The file includes the detailed targets of a drug. The format is similar with the existing file.

2. focused_gene.txt: This file lists the gene name related to the certain mechnisim of a disease, such as pyroptosis therapy in Triple-Negative Breast Cancer.

3. patient_info: These files require both the Overall Survival (OS) / Recurrence-Free Survival (RFS) data and the gene expression profiles for each patient. If the format is different, please moditify the function read_patient_info() in train.py.



If you have any questions, please contact Caihua Shan caihuashan@microsoft.com
