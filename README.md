# Code for Biological Factor Regulatory Neural Networks for Pyroptosis Therapy in Triple-Negative Breast Cancer

The current repository contains the source code for generating the ranking of drug combinations.


## Installation

The code has been tested with Python 3.9 on macOS 14.5 and includes libraries such as torch, torch_geometric, numpy, networkx, scikit-learn, among others.

pip install -r requirements.txt 

It typically takes a few minutes.

You also need to download some large files and put them in data folder:

Gene expression profiling of patients:
https://www.dropbox.com/scl/fi/p9kb71hk2zskgt9r4somm/tnbc2022_448s_gene_tpm_matrix.csv?rlkey=20cm2gqm95fp8p94ayvl2xf9i&st=gwodwzgi&dl=0

PPI networks:
https://www.dropbox.com/scl/fi/x2rzpu21gz1d2ao8puyfh/9606.protein.info.v11.5.txt?rlkey=zdf1fflv7l87oamdp9lddaq1k&st=yenupldn&dl=0

https://www.dropbox.com/scl/fi/4l96h51vob8j81qzb4wyb/9606.protein.links.full.v11.5.txt?rlkey=w602rcdj3zhq3bcblhirv8jdu&st=c2xh22p3&dl=0

## Running

cd code/

python main.py --drug1 XXX --drug2 XXX

You can provide the edge threshold for the protein-protein interaction network (--score 0.6), the names of the two drugs you'd like to analyze (--drug1 XXX --drug2 XXX), the different training epochs ([200,1000]) and so on. 

For one drug combination, it takes a few seconds to compute and obtain the result.

If you want to compute all pairs of drug combinations, please use run.sh

Please note that the code is also compatible with the CUDA version. 


## Tutorial on New Drug Combinations

To utilize the code for new drug combinations, it is essential to prepare the following files beforehand:

1. drug_target.txt: The file includes the detailed targets of a drug. The format is similar with the existing file.

2. focused_gene.txt: This file lists the gene name related to the certain mechnisim of a disease, such as pyroptosis therapy in Triple-Negative Breast Cancer.

3. patient_info: These files require both the Overall Survival (OS) / Recurrence-Free Survival (RFS) data and the gene expression profiles for each patient. If the format is different, please moditify the function read_patient_info() in train.py.


If you have any questions, please contact Caihua Shan caihuashan@microsoft.com
