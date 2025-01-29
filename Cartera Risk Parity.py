# Clear all variables in the global scope
globals().clear()


import pandas as pd
import numpy as np

#dasdas

df = pd.read_excel("")

"""# **LIMPIAR BASE DE DATOS PARA SOLO DEJAR NUMEROS**"""

df_clean = df.select_dtypes(include=['number'])

"""# **CONVERTIR EN RENTABILIDADES**"""

def_returns = df_clean.pct_change()

def_returns.tail()

"""# **SELECCIONAR SOLO LAS COLUMNAS QUE TENGAN TODOS LOS DATOS ENTRE DOS PERIDOS DE TIEMPO**"""


def set_dataset(dataframe, time_period_days):
    dataframe = dataframe.iloc[time_period_days:, :]
    dataframe = dataframe.dropna(axis=1, how = 'any')
    
    return dataframe  



def_returns_2000 = set_dataset(def_returns, 1)



# Global minimum variance portfolio
# Pages 7-8 of https://faculty.washington.edu/ezivot/econ424/portfolioTheoryMatrix.pdf
#-------------------------------------
def min_var(returns_array, MV_lambda):
    n_ticks = returns_array.shape[1]
    A = np.zeros([n_ticks+1, n_ticks+1])
    A[:-1, :-1] = 2*np.cov(returns_array.T)
    A[-1, 0:-1] = 1
    A[0:-1, -1] = 1
    z0 = np.zeros([n_ticks+1, 1])
    z0[-1] = MV_lambda
    b = np.zeros([n_ticks+1, 1])
    b[-1] = 1
    z = np.linalg.inv(A) @ b
    return z[:-1]

MV_lambda = 0.5

print(min_var(def_returns_2000, MV_lambda),sum(min_var(def_returns_2000, MV_lambda)))


import pandas as pd
import numpy as np
from scipy.cluster import hierarchy as sch

#-------------------------------------
# Hierarchical risk parity by Marcos Lopez de Prado
# This code only taken from de Prado's paper at https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2708678
#-------------------------------------
def getIVP(cov, **kargs):
    # Compute the inverse-variance portfolio
    ivp = 1./np.diag(cov) # taking np.sqrt performs poorly
    ivp /= ivp.sum()
    return ivp

def getClusterVar(cov, cItems):
    # Compute variance per cluster
    cov_ = cov.loc[cItems, cItems] # matrix slice
    w_   = getIVP(cov_).reshape(-1, 1)
    cVar = np.dot(np.dot(w_.T, cov_), w_)[0, 0] # taking np.sqrt performs poorly
    return cVar

def getQuasiDiag(link):
    # Sort clustered items by distance
    link = link.astype(int)
    sortIx = pd.Series([link[-1, 0], link[-1, 1]])
    numItems = link[-1, 3]  # number of original items

    while sortIx.max() >= numItems:
        sortIx.index = range(0, sortIx.shape[0]*2, 2)  # make space
        df0 = sortIx[sortIx >= numItems]  # find clusters
        i = df0.index
        j = df0.values - numItems
        sortIx[i] = link[j, 0]  # item 1
        df0 = pd.Series(link[j, 1], index=i + 1)
        sortIx = pd.concat([sortIx, df0])  # usamos pd.concat() en lugar de append
        sortIx = sortIx.sort_index()  # re-sort
        sortIx.index = range(sortIx.shape[0])  # re-index

    return sortIx.tolist()


def getRecBipart(cov, sortIx):
    # Compute HRP alloc
    w = pd.Series(1, index=sortIx)
    cItems = [sortIx]  # initialize all items in one cluster
    while len(cItems) > 0:
        cItems = [i[j:k] for i in cItems for j, k in ((0, len(i)//2),
                                                      (len(i)//2, len(i))) if len(i) > 1]  # bi-section
        for i in range(0, len(cItems), 2):  # parse in pairs
            cItems0 = cItems[i]  # cluster 1
            cItems1 = cItems[i+1]  # cluster 2
            cVar0 = getClusterVar(cov, cItems0)
            cVar1 = getClusterVar(cov, cItems1)
            alpha = 1 - cVar0/(cVar0+cVar1)
            w[cItems0] *= alpha  # weight 1
            w[cItems1] *= 1-alpha  # weight 2

    return w

def correlDist(corr):
    # A distance matrix based on correlation, where 0<=d[i,j]<=1
    # This is a proper distance metric
    dist = ((1-corr)/2.)**.5  # distance matrix
    return dist

def getHRP(cov, corr):
    # Construct a hierarchical portfolio
    corr, cov = pd.DataFrame(corr), pd.DataFrame(cov)
    dist = correlDist(corr)
    link = sch.linkage(dist, 'single')
    sortIx = getQuasiDiag(link)
    sortIx = corr.index[sortIx].tolist()
    # recover labels
    hrp = getRecBipart(cov, sortIx)
    return hrp.sort_index()


def Hierarchical_Risk_Parity(portoflio):

  cov = portoflio.cov()
  corr = portoflio.corr()
  hrp = getHRP(cov, corr)

  return hrp


print(Hierarchical_Risk_Parity(def_returns_2000),sum(Hierarchical_Risk_Parity(def_returns_2000)))

"""# **SHARPE RATIO - TRAYNOR RATIO**"""

from scipy.stats import t

def SharpeRatio(portoflio, Weights, Risk_Free_Rate):

  cov_matrix = portoflio.cov()
  portfolio_volatility = np.sqrt(np.dot(Weights.T, np.dot(cov_matrix, Weights)))

  mean_returns = np.mean(np.dot(portoflio, Weights))

  Sharpe_ratio = (mean_returns - Risk_Free_Rate) / portfolio_volatility


  # Estadisticamente significativo diferente de 0

  Stat_T_Sharpe_Ratio = Sharpe_ratio * np.sqrt(len(portoflio))

  p_value = 1 - t.cdf(abs(Sharpe_ratio), len(portoflio - 1))

  return Sharpe_ratio, p_value



Risk_Parity_Weights = Hierarchical_Risk_Parity(def_returns_2000)

SharpeRatio(def_returns_2000, Risk_Parity_Weights, 0.0001698997)[1]

a = SharpeRatio(def_returns_2000, Risk_Parity_Weights, 0.0001698997)

"""# **PROBAR COMBINACIONES DE ACTIVOS ALEATORIAMENTE + RISK PARITY + SHARPE -> GUARDA TODO Y LO ORDENA DE MAYOR A MENOR SHARPE**"""

def best_asset_allocation_Sharpe_Sorting(porfolio, Risk_Free_Rate, n_simulations):

  df = pd.DataFrame(columns=['Seed_Random_Number', 'Sharpe_Ratio', 'P_Value'])

  for i in range(n_simulations):

    random_seed = np.random.randint(100000000) # Obtengo un numero aleatorio
    np.random.seed(random_seed) #Meto el numero aleatorio en la semilla

    number_assets = np.random.randint(8, 19) # Numero de assets que voy a quitar de estudio - entre 8 y 18

    porfolio_loop = porfolio[np.random.permutation(porfolio.columns)] # Reordenamos las columnas de forma aleatoria

    # Eliminar las primeras x columnas con drop
    porfolio_loop_desire_assets = porfolio_loop.drop(columns=porfolio_loop.columns[:number_assets])

    Weights = Hierarchical_Risk_Parity(porfolio_loop_desire_assets)

    Sharpe_Ratio = SharpeRatio(porfolio_loop_desire_assets, Weights, Risk_Free_Rate)

    df.loc[i] = [random_seed, Sharpe_Ratio[0], Sharpe_Ratio[1]]

  df = df.sort_values(by='Sharpe_Ratio', ascending=False)

  return df


a = best_asset_allocation_Sharpe_Sorting(def_returns_2000, 0.0001698997, 100)

b = a

"""GET THE BEST SHARPE RATIO PORTFOLIO"""

from scipy.stats import t



a.iloc[0,0]

np.random.seed(int(a.iloc[0,0])) #Meto el numero aleatorio en la semilla

number_assets = np.random.randint(8, 19) # Numero de assets que voy a quitar de estudio - entre 8 y 18

porfolio_loop = def_returns_2000[np.random.permutation(def_returns_2000.columns)] # Reordenamos las columnas de forma aleatoria

# Eliminar las primeras x columnas con drop
porfolio_loop_desire_assets = porfolio_loop.drop(columns=porfolio_loop.columns[:number_assets])

Weights = Hierarchical_Risk_Parity(porfolio_loop_desire_assets)

SharpeRatio(porfolio_loop_desire_assets, Weights, 0.0001698997)[0]

print(SharpeRatio(porfolio_loop_desire_assets, Weights, 0.0001698997)[0], a.iloc[0,1])















