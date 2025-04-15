import copy
import numpy
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
from scipy.stats import invwishart
from scipy.stats import spearmanr
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix

from dataclasses import dataclass
from multipledispatch import dispatch

numpy.random.seed(1234)

############################################3
# Utilities from Ricardo's Notebook
def dataframe_categorize(df, cat_threshold):
    for column in df.columns:
        if df[column].nunique() <= cat_threshold:
            df[column] = df[column].astype(int).astype('category')

def cov_to_corr(cov_matrix):
    std_dev = numpy.sqrt(numpy.diag(cov_matrix))
    outer_std_dev = numpy.outer(std_dev, std_dev)
    corr_matrix = cov_matrix / outer_std_dev
    corr_matrix[numpy.diag_indices_from(corr_matrix)] = 1
    return corr_matrix

def logistic(x):
    return 1 / (1 + numpy.exp(-x))

def learn_xgb(input_dat, output_dat):
    if isinstance(output_dat.dtype, pd.CategoricalDtype):
        bst = xgb.XGBClassifier(n_estimators=10, max_depth=2, learning_rate=1, objective='binary:logistic', enable_categorical=True)
    else:
        bst = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=3, enable_categorical=True)
    bst.fit(input_dat, output_dat, verbose=False)
    return bst

def predict_xgb(bst, x_eval, y_dtype):
    if isinstance(y_dtype, pd.CategoricalDtype):
        return bstpredict_proba(x_eval)[:, 1]
    return bst.predict(x_eval)

################################################################################
# generator
@dataclass
class SyntheticBackdoorModel:
    cov_x: numpy.array    # Covariance of covariates#
    coeff_a: numpy.array  # Coefficients for logistic model of treatment on covariates
    coeff_y: numpy.array  # Coefficients for logistic model of outcome of treatment and covariates
    x_idx: str         # Column names of covariates
    a_idx: str         # Column name of treatment
    y_idx: str         # Column name of outcome

def simulate_synthetic_backdoor_parameters(num_x, rho, prob_positive):
    df = num_x + 1
    shape_mat = numpy.ones([num_x, num_x]) * rho
    shape_mat[numpy.diag_indices_from(shape_mat)] = 1
    cov_x = cov_to_corr(invwishart.rvs(df=df, scale=shape_mat*df))
    coeff_a = numpy.abs(numpy.random.normal(loc=0, scale=1/numpy.sqrt(num_x), size=num_x + 1))
    sign = numpy.random.uniform(size=num_x + 1)
    coeff_a[sign > prob_positive] = -coeff_a[sign > prob_positive]
    coeff_y = numpy.abs(numpy.random.normal(loc=0, scale=1/numpy.sqrt(num_x + 1), size=num_x + 2))
    sign = numpy.random.uniform(size=num_x + 2)
    coeff_y[sign > prob_positive] = -coeff_y[sign > prob_positive]
    a_idx = 'A'
    x_idx = ['X' + str(i) for i in range(num_x)]
    y_idx = 'Y'
    new_model = SyntheticBackdoorModel(cov_x=cov_x, coeff_a=coeff_a, coeff_y=coeff_y, x_idx=x_idx, a_idx=a_idx, y_idx=y_idx)
    return new_model

@dispatch(int, SyntheticBackdoorModel)
def simulate_from_backdoor_model(n, model):
    num_x = model.cov_x.shape[0]
    x = numpy.random.multivariate_normal(numpy.zeros(num_x), model.cov_x, n)
    a_prob = logistic(numpy.concatenate((numpy.ones([n, 1]), x), axis=1)@model.coeff_a)
    a = numpy.array(numpy.random.uniform(0, 1, size=n) <= a_prob, dtype=int).reshape(n, 1)
    y_prob = logistic(numpy.concatenate((numpy.ones([n, 1]), x, a), axis=1)@model.coeff_y)
    y = numpy.array(numpy.random.uniform(0, 1, size=n) <= y_prob, dtype=int).reshape(n, 1)
    dat = pd.DataFrame(numpy.concatenate([a, x, y], axis=1), columns=[model.a_idx] + model.x_idx + [model.y_idx])
    dataframe_categorize(dat, 2)
    return dat

@dispatch(int, SyntheticBackdoorModel)
def simulate_from_controlled_backdoor_model(n, model):
    num_x = len(model.x_idx)
    controlled_model = copy.deepcopy(model)
    controlled_model.coeff_a = numpy.zeros(num_x + 1) # Erase "edges" from X to A
    dat = simulate_from_backdoor_model(n, controlled_model)
    return dat

@dispatch(int, SyntheticBackdoorModel)
def confounder_column_keeper(keep_k, model):
    keep_x = numpy.argsort(numpy.abs(model.coeff_a[1:]))[-keep_k:]
    x_select = [model.x_idx[i] for i in keep_x]
    return x_select

@dispatch(pd.DataFrame, SyntheticBackdoorModel)
def get_cate(df_sample, model):
    n = df_sample.shape[0]
    x = df_sample[model.x_idx]
    a = numpy.zeros(n).reshape(n, 1)
    cate = numpy.zeros(n, dtype=numpy.float32)
    for i in [-1, 1]:
        e_y = logistic(numpy.concatenate((numpy.ones([n, 1]), x, a), axis=1)@model.coeff_y)
        cate = cate + e_y * i
        a = a + 1
    return cate

#################################################################
# XGB fitting
@dataclass
class XGBoostBackdoorModel:
    bst_propensity: xgb.XGBModel # Propensity score model
    bst_outcome: xgb.XGBModel    # Propensity score model
    std_outcome: float           # Standard deviation of outcome regression, if applicable
    x_idx: str                   # Column names of covariates
    a_idx: str                   # Column name of treatment
    y_idx: str                   # Column name of outcome
    empirical_df: pd.DataFrame   # Empirical distribution for generating covariates

def learn_backdoor_model(train, x_idx, a_idx, y_idx):
    x_train, a_train, y_train = train[x_idx], train[a_idx], train[y_idx]
    ax_train = pd.concat([a_train, x_train], axis=1)
    bst_propensity = learn_xgb(x_train, a_train)
    bst_outcome = learn_xgb(ax_train, y_train)
    if isinstance(y_train.dtype, pd.CategoricalDtype):
        std_outcome = None
    else:
        std_outcome = numpy.std(bst_outcome.predict(ax_train))
    return XGBoostBackdoorModel(bst_propensity=bst_propensity, 
                                bst_outcome=bst_outcome, 
                                std_outcome=std_outcome, 
                                x_idx=x_idx, a_idx=a_idx, y_idx=y_idx, 
                                empirical_df=train)

@dispatch(int, XGBoostBackdoorModel)
def xgboost_simulate_from_backdoor_model(n, model):
    row_choices = numpy.random.choice(range(model.empirical_df.shape[0]), size=n)
    df_sample = model.empirical_df.iloc[row_choices, :].reset_index(drop=True)
    propensities = model.bst_propensity.predict_proba(df_sample[model.x_idx])[:, 1]
    df_sample[model.a_idx] = numpy.random.binomial(n=1, p=propensities)

    e_y = predict_xgb(model.bst_outcome, df_sample[[*[model.a_idx], *model.x_idx]], model.empirical_df[model.y_idx])
    if model.std_outcome == None:
        df_sample[model.y_idx] = numpy.random.binomial(n=1, p=e_y)
    else:
        df_sample[model.y_idx] = e_y + numpy.random.normal(loc=0, scale=model.std_outcome, size=n)
    return df_sample

@dispatch(int, XGBoostBackdoorModel)
def xgboost_simulate_from_controlled_backdoor_model(n, model):
    row_choices = numpy.random.choice(range(model.empirical_df.shape[0]), size=n)
    df_sample = model.empirical_df.iloc[row_choices, :].reset_index(drop=True)
    df_sample[model.a_idx] = numpy.random.binomial(n=1, p=0.5 * numpy.ones(n))
    e_y = predict_xgb(model.bst_outcome, df_sample[[*[model.a_idx], *model.x_idx]], model.empirical_df[model.y_idx])
    if model.std_outcome == None:
        df_sample[model.y_idx] = numpy.random.binomial(n=1, p=e_y)
    else:
        df_sample[model.y_idx] = e_y + numpy.random.normal(loc=0, scale=model.std_outcome, size=n)
    return df_sample

@dispatch(int, XGBoostBackdoorModel)
def xgboost_confounder_column_keeper(keep_k, model):
    keep_x = numpy.argsort(model.bst_propensity.feature_importances_)[-keep_k:]
    x_select = [model.x_idx[i] for i in keep_x]
    return x_select

@dispatch(pd.DataFrame, XGBoostBackdoorModel)
def xgboost_get_cate(df_sample, model):
    n = df_sample.shape[0]
    a = pd.DataFrame(numpy.zeros([n, 1]), columns=[model.a_idx])
    ax_sample = pd.concat([a, df_sample[model.x_idx]], axis=1)
    cate = numpy.zeros(n, dtype=numpy.float32)
    for i in [0, 1]:
        e_y = predict_xgb(model.bst_outcome, ax_sample, df_sample[model.y_idx])
        cate = cate + (2 * i - 1) * e_y
        ax_sample[model.a_idx] = ax_sample[model.a_idx] + 1
    print(e_y[:20])
    return cate

def xgboost_get_cate(df_sample, model):
    n = df_sample.shape[0]
    a = pd.DataFrame(numpy.zeros([n, 1]), columns=[model.a_idx])
    ax_sample = pd.concat([a, df_sample[model.x_idx]], axis=1)
    cate = numpy.zeros(n, dtype=numpy.float32)
    for i in [0, 1]:
        #e_y = logistic(numpy.concatenate((numpy.ones([n, 1]), x, a), axis=1)@model.coeff_y)
        e_y = predict_xgb(model.bst_outcome, ax_sample, df_sample[model.y_idx])
        cate = cate + (2 * i - 1) * e_y
        ax_sample[model.a_idx] = ax_sample[model.a_idx] + 1
    print(e_y[:20])
    return cate

###################################################3
# cate and probability fitting
def trainXgb(train, test):
    x_train, a_train, y_train = train 
    x_test, a_test = test 
    ax_train = pd.concat([a_train, x_train], axis=1)
    ax_test = pd.concat([a_test, x_test], axis=1)
    
    bst_propensity = learn_xgb(x_train, a_train)
    train_p = bst_propensity.predict_proba(x_train)[:, 1]
    test_p = bst_propensity.predict_proba(x_test)[:, 1]
    
    bst_cate = learn_xgb(ax_train, y_train)
    train_c = bst_cate.predict(ax_train)
    test_c = bst_cate.predict(ax_test)
    return train_p, train_c, test_p, test_c


############################################################################
# The experiment is purely synthetic. The goal is to check whether removing covariates help improve the model performance on the observational (test) data.
# I generated a random model and produced synthetic data from it (no fitting of an extra model) using Ricardo's simulators
num_x = 50          # Number of covariates
rho = 0.2           # Hyperparameter controlling how (positively) correlated covariates are expected to be
prob_positive = 0.7 # Bias for positive coefficients
n = 50000           # Large sample size to understand the effect of a wrong model
max_n_rct = 1000        # Maximum allowed RCT sample
synth_model = simulate_synthetic_backdoor_parameters(num_x, rho, prob_positive)
df_sample_synth = simulate_from_backdoor_model(n, synth_model)

# The idea is to vary the number of covariates that are removed and check if the model performance on (unseen) observational data improves.
# The data are split using the restriction of Ricardo's Notebook
num_restrictions = 2
r_A, r_b = numpy.zeros([num_x + 2, num_restrictions]), numpy.zeros(num_restrictions)
r_A[numpy.where(df_sample_synth.columns == 'X1')[0][0], 0], r_b[0] = -1, -1 # X1 >= 1
r_A[numpy.where(df_sample_synth.columns == 'X2')[0][0], 1], r_b[1] = 1, 0   # X2 <= 0
drop_rct = numpy.where((df_sample_synth@r_A > r_b).sum(axis=1) > 0)[0]
drop_obs = numpy.where((df_sample_synth@r_A <= r_b).sum(axis=1) > 0)[0]
rct_sample = df_sample_synth.drop(drop_rct).reset_index(drop=True)
obs_sample = df_sample_synth.drop(drop_obs).reset_index(drop=True)
if rct_sample.shape[0] > max_n_rct:
  rct_sample = rct_sample.drop(range(max_n_rct, rct_sample.shape[0]))
# The two data sets have no overlap.
# The ground truth cate is measured on all data, and on the RCT and OBS separately.
true_cate_all = get_cate(df_sample_synth, synth_model)
true_cate_rct = get_cate(rct_sample, synth_model)
true_cate_obs = get_cate(obs_sample, synth_model)
# To make the comparison fair, we do not test the X-dropping models against the true CATE but against an XGBoost model of it fitted  on all data, i.e. a 'correct model' of the true cate.
alldata = df_sample_synth[synth_model.x_idx], df_sample_synth[synth_model.a_idx], true_cate_all
train = rct_sample[synth_model.x_idx], rct_sample[synth_model.a_idx], true_cate_rct
test = obs_sample[synth_model.x_idx], obs_sample[synth_model.a_idx]
# XGBoost is used to fit a model for CATE and a model for the probability of being in the RCT, given the covariates. To compare the performance of the X-dropping models on both the OBS and the RCT, we also run the obtained 'correct model' on the RCT and the OBS separately.
all_train_p, all_train_c, full_test_p, full_test_c = trainXgb(alldata, test)
all_train_p, all_train_c, full_train_p, full_train_c = trainXgb(alldata,[rct_sample[synth_model.x_idx], rct_sample[synth_model.a_idx]])
# We vary the number accepted covariates, ranked according to the feature-ranking method in the synthetic model (here probably we are slightly cheating)
tryK = range(len(synth_model.x_idx))
# For each number of accepted covariates, we measure the OBS and RCT Mean Square Error and the 'predicted probability of being in the RCT'
mse = []
mseTrain = []
proba = []
for k in tryK:
    # obtain the list of selected covariates
    keep_k = k
    x_select_synth = confounder_column_keeper(keep_k, synth_model)
    # prepare X-reduced training and testing samples
    train = rct_sample[x_select_synth], rct_sample[synth_model.a_idx], true_cate_rct
    test = obs_sample[x_select_synth], obs_sample[synth_model.a_idx]
    # fit a X-reduced model on the RCT data and test it on the OBS data 
    train_p, train_c, test_p, test_c = trainXgb(train, test)
    # compare the RCT and OBS cate predictions of the obtained X-reduced model with the predictions of the correct model (all covariates and all samples)
    mse.append(numpy.mean(abs(full_test_c  - test_c)))
    mseTrain.append(numpy.mean(abs(full_train_c - train_c)))
    # evaluate the average probability of being in the RCT of the observational users
    proba.append(numpy.mean(test_p))
    print('--------------------------------------------')
    print('k=', keep_k)
    print('incorrect model: X = ', x_select_synth)
    print('MSE for incorrect model:', mse[-1])
    print('min train_p, min test_p:', numpy.min(test_p), numpy.mean(test_p))
# Normalize the results to plot everything together                                
mse = [x.tolist()/max(mse) for x in mse[1:]]
mseTrain = [x.tolist()/max(mseTrain) for x in mseTrain[1:]]
proba = [x.tolist()/max(proba) for x in proba[1:]]
plt.plot(tryK[1:], mse, label='obs mse')
plt.plot(tryK[1:], mseTrain, label='rct mse')
plt.plot(tryK[1:], proba, label='$E_{OBS}(Prob(X \\in RCT))$')
plt.xlabel('number of covariates') 
plt.ylabel('normalized rct and obs mse or estimated RCT probability') 
plt.legend()
plt.show()


"""

a purely synthetic data set was generated using Ricardo's exact simulators 

synth_model = simulate_synthetic_backdoor_parameters(num_x, rho, prob_positive)
df_sample_synth = simulate_from_backdoor_model(n, synth_model)

with

num_x = 50
rho = 0.2
prob_positive = 0.7
n = 50000  
max_n_rct = 1000    

the synthetic data were then cut into OBS and RCT according to the restrictions




#################################################
def getMask(n):
    #M = numpy.array([[1 for i in range(n-j)]+[0 for i in range(j)] for j in range(n)])
    M = 1 - numpy.triu(numpy.ones([n, n])) + numpy.eye(n)
    return M

def createSets(X, theta):
    m = numpy.median(X @ theta)
    D1 = numpy.array([x for x in X if (x @ theta) > m])
    D2 = numpy.array([x for x in X if (x @ theta) <= m])
    return D1, D2

def fitLinear(X, y, rho):
    #w = MLPRegressor(hidden_layer_sizes=(1 + rho, 5), 
    #                 max_iter=100, tol=0.1, 
    #                 random_state=0)
    w = RandomForestRegressor(n_estimators=10 + 2 * rho, max_depth=2+len(X[0]))
    w.fit(X, y)
    print(2 + len(X[0]), 'done!')
    #M = X.T @ X + rho * numpy.eye(len(X.T))
    #w = numpy.linalg.pinv(M) @ X.T @ y
    return w

def linearPredict(X, w):
    return w.predict(X)
    #return X@w

def cate(X, W):
    Y = numpy.zeros(len(X))
    for iw in range(len(W[0])):
        #print(numpy.power(X, iw).shape)
        Y = Y +  numpy.power(X, iw) @ W[:, iw]
    return Y



N = 100
d = 20
K = 2
theta = [numpy.random.randn() for i in range(d)]
X = numpy.random.randn(N, d)
W = numpy.random.randn(d, K)
W = W * (abs(W) > numpy.median(abs(W)))


DX = createSets(X, theta)
DY = [cate(x, W) for x in DX]

labels = numpy.array(sum([[1-i for x in DX[i]] for i in [0, 1]], []))
features = numpy.array(DX[0].tolist() + DX[1].tolist())
print(len(labels))
print(len(features))
classifier = RandomForestClassifier(max_depth=2, random_state=0)
classifier.fit(features, labels)
f = numpy.array([x[1] for x in classifier.predict_proba(features)])
order = numpy.argsort(f)
f = f[order]
labels = labels[order]
features = features[order]
y = cate(features, W)



p = len(y)
tryRho = [10 + i for i in range(p)]
rctx = numpy.array([features[i] for i in range(len(f)) if labels[i] == 1])
rcty = [y[i] for i in range(len(f)) if labels[i] == 1]
featureSelection = [numpy.random.choice(len(rctx[0]), 1 + int(numpy.floor(i * d /p))) for i in range(p)]
models = [fitLinear(rctx[:, featureSelection[j]], rcty, tryRho[j]) for j in range(p)]
roofmodel = fitLinear(features, y, p)
roofline = linearPredict(features, roofmodel)
fits = [linearPredict(features[:, featureSelection[i]], models[i]) for i in range(len(models))]
fits = numpy.array(fits).T
mask = getMask(len(fits))
#complete = noisy_matrix_completion(fits, mask, lambd=1.0, max_iters=500)

SVD = numpy.linalg.svd(fits)
s = SVD.S/sum(SVD.S)
r = max([i for i in range(len(s)) if sum(s[:i]) < .7])
print('r=',r)
complete = SVD.U[:, :r] @ numpy.diag(SVD.S[:r]) @ SVD.Vh[:r, :]
#complete = complete * mask


trues = [y for i in range(len(fits))]
trues = numpy.array(trues).T
errorOriginal = abs(numpy.array(fits) - numpy.array(trues)) * mask
errorSmooth = abs(numpy.array(complete) - numpy.array(trues)) * mask

#complete = complete*mask
extrapolate = complete[:, -1]
#extrapolate = numpy.diag(complete)
baseline = fits[:, -1]
true = trues[:, 0]
q = [extrapolate, baseline, roofline, true]
q = [numpy.array([x[i] for i in range(len(x)) if labels[i]==0]) for x in q]
ee = [numpy.round(numpy.mean(abs(x-q[-1])), 3) for x in q]

names = ['completion', 'basline', 'roofline', 'true']
colors = ['r*', 'b*', 'g*', 'k.']
for i in range(len(names)):
    plt.plot(q[-1], q[i], colors[i], label=names[i] + ', MSE='+str(ee[i]))
plt.xlabel('true')
plt.ylabel('predicted')
plt.legend()
plt.show()

# Create subplots
fig, axs = plt.subplots(1, 2, figsize=(10, 5))  # 1 row, 2 columns
# First matshow
axs[0].matshow(errorOriginal, cmap='viridis')
axs[0].set_title('original matrix')
axs[0].set_xlabel('$\\longleftarrow \\rho $')
axs[0].set_ylabel('$\\longleftarrow \\hat {\\rm Prob}(X \\in  {\\rm RCT}) $')

# Second matshow
axs[1].matshow(errorSmooth, cmap='viridis')
axs[1].set_title('completed matrix')
axs[1].set_xlabel('$\\longleftarrow \\rho $')
axs[1].set_ylabel('$\\longleftarrow\\hat {\\rm Prob}(X \\in  {\\rm RCT})$')

# Adjust layout
plt.tight_layout()
plt.show()




#full = RandomForestRegressor(max_depth=2, random_state=0)
#full = MLPRegressor(hidden_layer_sizes=10, max_iter=2000, tol=0.1, random_state=0)
X = numpy.concatenate(DX, axis=0)
Y = numpy.concatenate(DY, axis=0)
#full.fit(X, Y)
#Yf = full.predict(X)
wfull = fitLinear(X, Y, 0)
Yf = linearPredict(X, wfull)

#rct = RandomForestRegressor(max_depth=2, random_state=0)
#rct = MLPRegressor( hidden_layer_sizes=10, max_iter=2000, tol=0.1, random_state=0)
#rct.fit(DX[0], DY[0])
#Yrct = rct.predict(X)
wrct = fitLinear(DX[0], DY[0], 0)
Yrct = linearPredict(X, wrct)

sets = ['rct', 'obs']
markers = ['*', 'o']
for iD in range(len(DX)):
    x = DX[iD]
    y = DY[iD]
    #yf = full.predict(x)
    yf = linearPredict(x, wfull)
    #yrct = rct.predict(x)
    yrct = linearPredict(x, wrct)
    plt.plot(y, yf, markers[iD]+'g', label=sets[iD]+' users-full model')
    plt.plot(y, yrct, markers[iD]+'r', label=sets[iD]+' users-rct model')
plt.plot([min(Y), max(Y)], [min(Y), max(Y)], '--k')
plt.legend()
plt.show()

n = len(DX[0]) + len(DX[1])
rhoMin, rhoMax = 0, 1
tryRho = [rhoMin + i * (rhoMax-rhoMin)/(n+1) for i in range(n)]
models = [fitLinear(DX[0], DY[0], rho) for rho in tryRho]
fits = [linearPredict(X, w) for w in models]
mask = numpy.triu(numpy.ones([n, n]))
matrix = fits * mask
print(matrix[:5, :5])
SVD = numpy.linalg.svd(fits)
r = 2
print(SVD.S[:10])
complete = SVD.U[:, :r] @ numpy.diag(SVD.S[:r]) @ SVD.Vh[:r, :]
complete = complete * mask
print(complete[:5, :5])




"""

