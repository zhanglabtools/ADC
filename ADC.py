import numpy as np
from math import *
import sklearn
from dcorT import *
from scipy import stats
import pandas as pd
import statsmodels.stats.multitest as multi
from tqdm import tqdm_notebook

def sim_gene(X_cor, gene_list, n_approx = 30):
    '''
    This function selects approximated observations for each gene.
    ----------------------------------------------------------------------
    Parameters
    -----------
    X_cor: The correlation matrix for cells in dataX. 
    TYPE: numpy array.
    shape: (n_cells, n_cells).
    ----
    gene_list: The genes interstec1d in dataX and dataY.
    TYPE: lisr or numpy array.
    ----
    n_approx: the number of approximated observations.
    TYPE : int
    ----------------------------------------------------------------------
    Returns
    -----------
    The dict with keys: genes interstec1d in dataX and dataY and
    values: the approximations for the target gene. 
    ----------------------------------------------------------------------
    '''
    if not isinstance(n_approx, int):
        return print('The number of the approximated observations should be integer.')


    gene_list = np.array(gene_list)
    similar_genes = n_approx
    similar_gene_dict = {}
    
    nearest = np.argsort(-1*X_cor, axis=1)
    for i in range(nearest.shape[0]):
        Xi_largest_cor = nearest[i, 0:similar_genes]
        similar_gene_dict[gene_list[i]] = list(gene_list[Xi_largest_cor])
            
    return similar_gene_dict



def ADC(X, Y, FDR=0.05, ensemble=False):
    '''
    This function calculates similarly expressed genes passed FDR contorl.
    ----------------------------------------------------------------------
    Parameters
    -----------
    X: The gene expression matrix for data X. 
    TYPE: pd.DataFrame.
    shape: (n_genes，n_cells).
    ----
    Y: The gene expression matrix for data Y. 
    TYPE: pd.DataFrame.
    shape: (n_genes，n_cells).
    ----
    FDR: The Fasle Discovery Rate.
    TYPE: float.
    ----
    ensemble: Whether to use the ensemble version of ADC.
    TYPE: boolean.    
    ----------------------------------------------------------------------
    Returns
    -----------
    Similarly expressed genes whcih pass the FDR contorl. 
    ----------------------------------------------------------------------
    '''
    X_genes = X.index
    Y_genes = Y.index

    #calculate the common genes
    gene_list = np.intersect1d(X_genes, Y_genes) 
    X1 = X.loc[gene_list, :]
    Y1 = Y.loc[gene_list, :]
    
    #Calculate the pearson correlation matrix to prepare for the selecting of approximated observations
    X_cor = np.corrcoef(X1)
    Y_cor = np.corrcoef(Y1)
    
    if not ensemble:
        #select approximated observations for each gene
        X_similar_gene_dict = sim_gene(X_cor, gene_list)
        Y_similar_gene_dict = sim_gene(Y_cor, gene_list)
        
        W = []#store the p-value of distance correlation coefficient
        for X_gene in gene_list:
            #X_rep stores expressions of  approximated observations for each targe gene in data X
            X_rep = np.array(X.loc[X_similar_gene_dict[X_gene], :])
            Y_rep = np.array(Y.loc[Y_similar_gene_dict[X_gene], :])
            
            #calculate the p-value of distance correlation coefficient
            cor_org = dcorT_p(X_rep, Y_rep)
        
            W.append(cor_org)
        

        #FDR control    
        W_FDR = multi.multipletests(W, alpha=FDR, method='fdr_bh')[0]
        
        #store similar expressed genes
        simialr_genes = []
        for i in range(len(W_FDR)):
            if W_FDR[i]:
                simialr_genes.append(gene_list[i])
        return simialr_genes
        
    else:
        #store similarly expressed genes for all the $k$s
        all_gene_list = []
        for n_similar_genes in tqdm_notebook(range(20,41)):
            
            #select approximated observations for each gene
            X_similar_gene_dict = sim_gene(X_cor, gene_list, n_approx = n_similar_genes)
            Y_similar_gene_dict = sim_gene(Y_cor, gene_list, n_approx = n_similar_genes)

            #store the p-value of distance correlation coefficient       
            W = []
            for X_gene in gene_list:
                #X_rep stores expressions of  approximated observations for each targe gene in data X
                X_rep = np.array(X.loc[X_similar_gene_dict[X_gene], :])
                Y_rep = np.array(Y.loc[Y_similar_gene_dict[X_gene], :])
                
                #calculate the p-value of distance correlation coefficient
                cor_org = dcorT_p(X_rep, Y_rep)

                W.append(cor_org)
                
            #FDR control
            W_FDR = multi.multipletests(W, alpha=FDR, method='fdr_bh')[0]
            simialr_genes = []
            for i in range(len(W_FDR)):
                if W_FDR[i]:
                    simialr_genes.append(gene_list[i])
            all_gene_list.extend(simialr_genes)

        #count the total frequency of all the genes selected by ADC 
        #under different $k$, i.e. number of approximated observations.
        gene_set = set(all_gene_list)
        gene_dict = {}
        for gene in all_gene_list:
            if gene in gene_dict.keys():
                gene_dict[gene] += 1
            else:
                gene_dict[gene] = 1

        select_genes = []
        for gene in gene_dict.keys():
            # Only select genes selected by > 10 different $k$s
            if gene_dict[gene] >= 10:
                select_genes.append(gene)
        return select_genes
        
    
    