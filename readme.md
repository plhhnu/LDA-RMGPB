# Deciphering lncRNA-disease associations based on multi-representation fusion and boosting with Gaussian process

LncRNA-disease association (LDA) identification can provide valuable insights for understanding disease pathogenesis. Existing most deep learning-based LDA prediction models remain limitations in effectively fusing various features of lncRNAs and diseases and accurately classifying unknown lncRNA-disease pairs (LDPs). Here, we introduce a deep learning-based LDA prediction framework named LDA-RMGPB based on multi-representation fusion and boosting with Gaussian process. First, a randomized singular value decomposition model is presented to extract LDP linear features. Subsequently, a masked graph autoencoder is exploited to learn LDP nonlinear features. Finally, a boosting algorithm with Gaussian process takes the concatenation of LDP linear and nonlinear features as inputs and classifies unlabeled LDPs. To measure the LDA-RMGPB performance, we performed a series of experiments. Using six evaluation metrics, under four different 5-fold cross-validation strategies (i.e., cross validations on lncRNAs, diseases, LDPs, independent lncRNAs and independent diseases), LDA-RMGPB greatly surpassed seven state-of-the-art prediction methods on two LDA datasets. Further analysis, including ablation study, CeRNA theory analysis, lncRNA-related therapeutic drug analysis, and survival analysis, elucidated that LDA-RMGPB achieved superior LDA identification ability. Moreover, we predicted that lncRNAs ATP6V1G2-DDX39B and PSORS1C3 could have dense linkages with breast cancer and prostatic neoplasms, respectively. We anticipate that LDA-RMGPB contributes to the discovery of novel therapeutic molecular targets across diverse diseases. LDA-RMGPB is freely available at https://github.com/plhhnu/LDA-RMGPB.

## 1. Flowchart

![Figure 1:The flowchart of LDA-RMGPB](flowchart.pdf)

## 2. Running environment

```
python version 3.10.12
numpy==1.22.4
pandas==2.1.1
scikit-learn==1.3.1
scipy==1.11.3    
torch==2.1.0+cu118
torch-cluster==1.6.3+pt21cu118
torch-geometric==2.4.0
torch-scatter==2.1.2+pt21cu118
torch-sparse==0.6.18+pt21cu118
gpboost==1.5.1
```

## 3. Data

```
In this work，lncRNADisease is data 1 and MNDR is data 2.
```

## 4. Usage

Default is 5-fold cross validation from four strategy (ie.S1, S2, S3, and S4) on lncRNADisease and MNDR databases. To run this model：

```
python  3 Gussian process boosting/3 GPBoost/main.py
```

Extracting linear features of lncRNAs and diseases by randomized SVD, to run:

```
python  1 Randomized SVD/SVD.py
```

Extracting non-linear features of lncRNAs and diseases by masked GAE, to run:

```
python  2 Masked GAE/main_justTrain.py
```

## 5. Details

We calculate lncRNA functional similarity and disease semantic similarity using the IDSSIM approach.(Fan, W., Shang, J., Li, F. *et al.* IDSSIM: an lncRNA functional similarity calculation model based on an improved disease semantic similarity method. *BMC Bioinformatics* **21**, 339 (2020). https://doi.org/10.1186/s12859-020-03699-9)
