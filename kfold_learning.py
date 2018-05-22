import os
import sys
import pandas
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from copy import deepcopy
import numpy as np
import scipy.stats as stats
from sklearn import preprocessing, linear_model
from sklearn.model_selection import KFold
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils import check_array

def kfold_feature_learning(train, test, y, t_y, clf = linear_model.LassoCV(cv=10), problem = 'regression', 
                           folds = 10, scale=True, verbose = True, search = False, shuffle_k = False, ci = None,
                           p_cutoff = None, regcols = None, regdf = None, keep_cols = None, 
                           out_dir = None,  output='light', save_int = True, vote = False, weighted = False,
                           hide_test = False):
    '''
    This is a function that will use nested cross validation to generate an average model
    that will hopefully generalize better to unseen test data. 
    
    You can must enter your training and testing data, and your y variable for both, the model you 
    wish to use for prediction, and whether the problem is classification or regression.
    
    The function will run K iterations of prediction on your training set, and will average 
    the weights across folds for a final model. The final model will then be applied to your 
    testing data. The validation and testing accuracy will be displayed. 
    
    Several other options exists (see below), and may be forthcoming.
    
    ATTENTION: THIS FUNCTION IS STILL IN DEVELOPMENT. IT IS UGLY AND UNFINISHED, SO DONT JUDGE!
    SOME FEATURES ARE IN BETA, AND IT WILL NOT ACCEPT SOME MODELS FOR CLF!
    
    
    *** USER-DEFINED ARGUMENTS ***
    
    -- train is a subjects x variables dataframe (this represents your training data)
    -- y is a pandas Series with the same index as train. y should not be in train
    
    # NOTE: train and test indices should just be a range
    
    -- test is a subjects x variables dataframe (this represents your independent test data)
    -- t_y is a pandas seris with the same index as test. y should not be in test

    
    *** MODEL OPTIONS ***

    -- clf: here, you can enter in whatever model you want with whatever parameters you want.
    
    -- if your model (clf) is a regression model (e.g. Lasso, SVR), leave problem as "regression". 
    If it is a classification model (e.g. SVM, SGD, etc.), change problem to "classification"
    
    -- folds: how many fold cross-validation should occur within the outer loop of the
    training dataset. Less folds means less trained models, but with more different data.
    
    -- scale: if True, train will be scaled with a Standard Scaler, and test will be transformed 
    to this scale
    
    -- verbose: if you do not want any output (including scores at the end!!), set this to False.
    
    -- search: if clf is a model_selector (such as GridSearch), MAKE SURE you set this to True,
    or the script will fail.

    -- shuffle_k: Do you want to shuffle your data before splitting into folds? If left to None,
    data will split into folds without shuffling (this could be dangerous if your dataset is,
    for example, sorted by your y variable). Setting this argument to an integer will shuffle
    the data before splitting, and the integer will determine the random state.

    -- ci: if set to a float between 0 and 1, confidence intervals around the prediction scores
    will be calculated using bootstrapping. For example, a value of 0.95 would create 95% 
    confidence intervals around the accuracy estimate.
    
    
    *** FEATURE SELECTION OPTIONS ***
    
    -- p_cutoff: if you wish to only keep features statistically related to y (through t-test 
    or correlation), you can control the alpha value here. Leave as None to use all features
    
    -- reg_cols: a list of column labels in regdf. All labels specified in this list will be 
    regressed out of all other model features during "feature selection" (i.e. when features are
    removed via the p_cutoff argument). In other words, this argument can be used if you only
    want to include features in your model that are significant when adjusting for the variables
    specified in reg_cols. Leave as None if you don't want this argument active.
    
    -- regdf: a subject x variables pandas Dataframe that contain anything as long as it has all 
    (matched) indices in train, and all columns in reg_cols
    
    -- keep_cols: a list of column labels. These are variables in train that you wish to retain 
    in your model no matter what, even if they do not pass the feature selection. For example, if
    age is not selected by your p_cutoff, but you still want age in your model, you can list in
    keep_cols
    
    *** OUTPUT OPTIONS ***
    
    -- out_dir: will save your weight matrix and predicted values to a directory you specify
    
    -- output: decide what you want the function to return:
        * 'scores' will only return the r2 (regression) or sensitivity, specificity and accuracy
            (classification) of you validation and test.
        * 'ci' will return the confidence intervals around your model fit (if CI argument is set)
        * 'light' will return the weights of your final model, the predicted values of your
            validation, the predicted values of your test, and the intercept of the final model, 
            in that order.
        * 'heavy' if problem = regression, will return everything from light, plus a matrix 
            containing weights from all folds of the validations. Will also return all models 
            from all folds.
            if problem = classification, will return a summary dataframe (with weights) for your
            validation, a summary dataframe for your test, your predicted values from validation,
            predicted values from test, a matrix containing weights from all folds of the 
            validation, and the model from the most recent fold.
    
    *** PREDICTION OPTIONS ***

    -- save_int: if for some reason you don't want to apply the intercept to your final predicted 
        values, set this to False. Currently not supported if 'vote' argument passed

    -- vote: A different way of predicting. Instead of averaging the weights from several folds,
        predicts values using the model from each fold and averages solution. For classification,
        amnbiguous cases (i.e ties across models) are identified. This is currently the only 
        way this function can support random forests.
            * 'hard' will us the predict function and average the responses
            * 'soft' will us the predict_proba and average the responses
            * None will not use the vote method

    -- weighted: If True, model coefficients (weights) for each fold of the outer loop will be 
        themselves weighted by the prediction accuracy *within* that fold. Or, if vote is not None,
        predictions votes will be weighted by within-fold accuracy scores. In this way, models
        produced by folds with worse prediction accuracy within their fold will contribute less to
        the overall model.

    -- hide_test: If True, the testing accuracy will not be calculated or revealed. Using this
    option, only validation accuracy will be calculated and listed. This is a good option for
    tweaking inputs before burning your test data.

        '''

    # check inputs
    if problem != 'regression' and problem != 'classification':
        raise IOError('please set problem to regression or classification')

    if hasattr(clf, 'max_leaf_nodes'):
        print('random forest detected, setting vote to \'soft\'')
        vote = 'soft'
        save_int = False
        all_ints = np.nan

    if vote and weighted and problem == 'regression':
        print('WARNING: vote does not work with weighting for regression problem')
        print('changing vote to None')
        vote = None
    #feature_matrix = pandas.DataFrame(np.zeros_like(train))
    
    if type(keep_cols) != None and type(keep_cols) != list:
        try:
            keep_cols = list(keep_cols)
        except:
            pass

    # Initiate variables
    predicted = []
    all_weights = pandas.DataFrame(np.zeros((folds,len(train.columns))))
    if save_int:
        ints = []
    start = 0
    fold = 1
    fail = False
    if vote:
        all_mods = {}
        save_int = False
        all_ints = np.nan
    else:
        all_mods = []

    # scale inputs
    if scale:
        master_scl = preprocessing.StandardScaler().fit(train)
        train = pandas.DataFrame(master_scl.transform(train),
                                 index=train.index,columns=train.columns)
        test = pandas.DataFrame(master_scl.transform(test),
                                 index=test.index,columns=test.columns)
    
    # strip columns names
    tr_cols = train.columns 
    train.columns = range(len(train.columns))
    test.columns = range(len(test.columns))
    if type(keep_cols) == list:
        tmpo = []
        for col in keep_cols:
            tmpo.append(tr_cols.tolist().index(col))
        keep_cols = tmpo
    ## ALSO DO FOR REGCOLS?

    # define kfold split
    if type(shuffle_k) == int:
        kfoldsplit = KFold(n_splits=folds, shuffle=True, random_state = shuffle_k).split(train)
    else:
        kfoldsplit = KFold(n_splits=folds).split(train)

    for tr_ix, te_ix in kfoldsplit:
        tmp_mtx = train.loc[train.index[tr_ix]] # working matrix
        
        # Build regression statements (if regcols)
        if regcols != None: 
            ref = deepcopy(tmp_mtx)
            tmp_mtx.columns = ['x_%s'%x for x in tmp_mtx.columns]
            tmp_mtx['y'] = y.loc[tmp_mtx.index]
            stmnt = 'y ~'
            for z,col in enumerate(regcols):
                cov = 'cov_%s'%z
                tmp_mtx[cov] = regdf.loc[tmp_mtx.index][col]
                if z == 0:
                    stmnt += ' %s'%cov
                else:
                    stmnt += ' + %s'%cov
        else:
            regcols = []
        
        # feature selection -- only retain significant features
        ps = []
        if p_cutoff != None:
            if len(regcols) > 0:
                if verbose:
                    print('running regression for fold %s of %s'%(fold,folds))
                for x in range(tmp_mtx.shape[1] - (len(regcols) + 1)):
                    n_stmnt = '%s + x_%s'%(stmnt,x)
                    ps.append(smf.ols(stmnt,data=temp_mtx).fit().pvalues[-1])
                sig_mtx = ref.loc[ref.index[:]]
            else:
                if problem == 'regression':
                    if verbose:
                        print('running correlation for fold %s of %s'%(fold,folds))
                    for x in range(tmp_mtx.shape[1]):
                        ps.append(stats.pearsonr(
                                y[tmp_mtx.index].values,tmp_mtx.values[:,x])[1]
                             )
                else: # classification
                    if verbose:
                        print('running ttests for fold %s of %s'%(fold,folds))
                    for x in range(tmp_mtx.shape[1]):
                        ps.append(stats.ttest_ind(
                                tmp_mtx.loc[y[tmp_mtx.index][y[tmp_mtx.index]==0].index][tmp_mtx.columns[x]],
                                tmp_mtx.loc[y[tmp_mtx.index][y[tmp_mtx.index]==1].index][tmp_mtx.columns[x]]
                             )[1])
            ps_s = pandas.Series(ps)
            sig = ps_s[ps_s < p_cutoff]
            if len(sig) == 0:
                fold += 1
                continue
            sig_mtx = tmp_mtx[sig.index]
        else:
            sig_mtx = tmp_mtx[tmp_mtx.columns[:]]
        
        # run model
        if verbose:
            print('running model for fold %s of %s'%(fold,folds))
        if type(keep_cols) == list:
            for col in keep_cols:
                sig_mtx[col] = tmp_mtx.ix[:,col]
        if search:
            mod_sel = clf.fit(sig_mtx,y[sig_mtx.index])
            new_clf = mod_sel.best_estimator_
            model = new_clf.fit(sig_mtx,y[sig_mtx.index])
        else:
            model = clf.fit(sig_mtx,y[sig_mtx.index])          
        if hasattr(model, 'coef_'):
            try:
                all_weights.loc[(fold-1)][sig_mtx.columns] = model.coef_
            except:
                try:
                    all_weights.loc[(fold-1)][sig_mtx.columns] = model.coef_[0,:]
                except:
                    print('error on fold %s'%(fold))
                    fold += 1
                    if len(regcols) == 0:
                        regcols = None
                    fail = True    
                    continue
        elif hasattr(model, 'feature_importances_'):
            all_weights.loc[(fold-1)][sig_mtx.columns] = model.feature_importances_
        else:
            raise AttributeError('this script does not currently support the model you entered')
        
        if weighted:
            scr_wt = model.score(train.loc[train.index[te_ix]][sig_mtx.columns].values, 
                        y[train.index[te_ix]].values)
            all_weights.loc[(fold-1)][sig_mtx.columns] = all_weights.loc[(fold-1)][sig_mtx.columns] * scr_wt

        # save predicted values for this validation fold
        [predicted.append(x) for x in model.predict(train.loc[train.index[te_ix]][
                                                    sig_mtx.columns].values)]
        if save_int:
            ints.append(model.intercept_)
        
        if vote:
            if weighted:
                all_mods.update({model: [sig_mtx.columns, scr_wt]})
            else:
                all_mods.update({model: sig_mtx.columns})
        else:
            all_mods.append(deepcopy(model))
        # reset variables
        fold += 1
        if len(regcols) == 0:
            regcols = None
        
        # save output
        if out_dir != None and type(out_dir) == str:
            print('saving matrix for fold %s of %s'%(fold,folds))
            if not os.path.isdir(out_dir):
                os.mkdir(out_dir)
            feature_matrix.to_csv(os.path.join(out_dir,'lasso_weights.csv'))
            pandas.DataFrame(pandas.Series(predicted)).to_csv(
                                                    os.path.join(out_dir,'lasso_predicted.csv'))
        
    # assemble final model
    final_weights = all_weights.mean(axis=0)
    n_feats = len([i for i in final_weights.index if abs(final_weights[i]) > 0 ])
    if verbose:
        print(n_feats,'features selected')
    
    if n_feats == 0 or fail == True:
        val_res, t_res = np.nan, np.nan
        predicted, t_predicted = [], np.array([])
        if save_int:
            all_ints = np.mean(ints)
        else:
            all_ints = np.nan
        val_sum, t_sum = pandas.DataFrame(), pandas.DataFrame()
    else:
    
        # run validation
        if problem == 'regression':
            if len(y[train.index]) != len(predicted):
                print('WARNING: No features selected in at least one fold')
                print('Try changing clf parameters, reducing p_cutoff, or getting some better data')
                val_res, t_res = np.nan, np.nan
                if type(ci) == float:
                    ci_l, ci_u, cim, p, r = np.nan, np.nan, np.nan, np.nan, np.nan
            else:
                r,p = stats.pearsonr(y[train.index],predicted)
                val_res = (r**2)*100
                if type(ci) == float:
                    distr = []
                    prd = pandas.Series(predicted,index=train.index)
                    for c in range(1000):
                        rsamp = np.random.choice(train.index, len(train.index))
                        distr.append(stats.pearsonr(y[rsamp], prd[rsamp])[0]**2)
                    nci = int(ci*1000)
                    ci_u = sorted(distr)[1000 - nci]
                    cim = np.mean(distr)
                    ci_l = sorted(distr)[(nci)]

            if verbose:
                if type(ci) == float:
                    print('validation prediction accuracy is %s percent (%s, %s, mean=%s) \n p = %s \n r = %s'%(val_res,ci_l, 
                                                                                                        ci_u, cim, p,r))
                else:
                    print('validation prediction accuracy is %s percent \n p = %s \n r = %s'%(val_res,p,r))
        else: # classification
            val_sum, val_res = manual_classification(y[train.index],predicted,verbose,'validation')

        # apply model to test data
        if hide_test:
            t_res = np.nan
            t_predicted, all_ints = [],[]
            if problem == 'classification':
                t_sum = np.nan
        else:
            if vote:
                t_predicted = vote_prediction(test, all_mods, problem, vote, weighted)
            else:
                ntest = check_array(test,accept_sparse='csr')
                t_predicted = pandas.Series(safe_sparse_dot(ntest,np.array(final_weights).T,dense_output=True),index=test.index)
                if save_int:
                    all_ints = np.mean(ints)
                    t_predicted += all_ints
                else:
                    all_ints = []

            # run test
            if problem == 'regression':
                r,p = stats.pearsonr(t_y[test.index],t_predicted)
                t_res = (r**2)*100
                if type(ci) == float:
                    distr = []
                    for c in range(1000):
                        rsamp = np.random.choice(test.index, len(test.index))
                        distr.append(stats.pearsonr(t_y[rsamp], t_predicted[rsamp])[0]**2)
                    nci = int(ci*1000)
                    ci_u = sorted(distr)[1000 - nci]
                    cim = np.mean(distr)
                    ci_l = sorted(distr)[(nci)]

                if verbose:
                    if type(ci) == float:
                        print('testing prediction accuracy is %s percent (%s, %s, mean=%s) \n p = %s \n r = %s'%(t_res,ci_l, 
                                                                                                                ci_u, cim, p,r))
                    else:
                        print('testing prediction accuracy is %s percent \n p = %s \n r = %s'%(t_res,p,r))
            else: # classification
                if not vote:
                    t_decision_func = t_predicted
                    t_predicted = pandas.Series(index = test.index)
                    t_predicted[t_decision_func[t_decision_func<0].index] = 0
                    t_predicted[t_decision_func[t_decision_func>0].index] = 1
                else:
                    t_decision_func = None
                t_sum, t_res = manual_classification(t_y[test.index],t_predicted,verbose,'testing',t_decision_func)

    # prepare outputs
    
    final_weights.columns = tr_cols
    all_weights.columns = tr_cols
    
    if output == 'scores':
        to_return = dict(zip(['validation_accuracy','test_accuracy'],
                            [val_res,t_res]))
    elif output == 'ci':
        labs = ['%dCI_%s'%(ci,x) for x in ['mean','upper','lower']] + ['distribution']
        to_return = dict(zip(labs, [cim, ci_u, ci_l, distr]))
    elif output == 'light':
        labs = ['validation_accuracy', 'test_accuracy', 'final_model_weights',
                'validation_predicted', 'test_predicted', 'model_intercepts']
        to_return = dict(zip(labs,
                            [val_res,  t_res, final_weights, predicted, t_predicted, all_ints]))
    else:
        if problem == 'regression':
            labs = ['final_model_weights', 'validation_predicted', 'test_predicted',
                    'model_intercepts', 'all_model_weights', 'all_models']
            to_return = dict(zip(labs,
                                [final_weights, predicted, t_predicted, all_ints, all_weights, all_mods]))
        else:
            labs = ['validation_summary', 'test_summary', 'validation_predicted', 'test_predicted',
                    'model_intercepts', 'all_model_weights', 'all_models']
            to_return = dict(zip(labs,
                                [val_sum, t_sum, predicted, t_predicted, all_ints, all_weights, all_mods]))
    return to_return

def manual_classification(obs, pred, verbose, mode='validation', weights=None):
            
    if type(obs) == pandas.core.series.Series:
        obs = obs.values
    
    if type(pred) == pandas.core.series.Series:
        pred = pred.values
    
    summary = pandas.DataFrame(index=range(len(obs)),columns = ['Predicted','Actual'])
    summary['Predicted'] = pred
    summary['Actual'] = obs
    if type(weights) != type(None):
        summary['Prediction Function'] = weights
    for x in summary.index: 
        if summary.ix[x,'Predicted'] == summary.ix[x,'Actual']:
            summary.ix[x,'Hit'] = 1
        else:
            summary.ix[x,'Hit'] = 0

    tp,tn,fp,fn = [],[],[],[]
    for i,row in summary.iterrows():
        val = row['Predicted'] - row['Actual']
        if val == 0:
            if row['Actual'] == 1:
                tp.append(i)
            else:
                tn.append(i)
        elif val == 1:
            fp.append(i)
        elif val == -1:
            fn.append(i)
        else:
            print('something went wrong for ',i)

    sens = len(tp)/((len(tp)+len(fn)) + 1e-06)
    spec = len(tn)/((len(tn)+len(fp)) + 1e-06)
    acc = (len(tp)+len(tn))/(len(tp)+len(fn)+len(tn)+len(fp))

    if verbose:
        print(mode,' sensitivity:' , sens)
        print(mode,'specificity:' , spec)
        print(mode,'accuracy:', acc)

    results = [sens,spec,acc]

    return summary, results

def vote_prediction(X_test, all_mods, problem, vote_type, weighted = False):

    if problem == 'classification':
        if not any(
                [hasattr(x, 'decision_function') for x in all_mods]
                ) and vote_type == 'hard':
            print('changing vote to soft out of necessity')
            vote_type = 'soft'
        
        preds = pandas.DataFrame(index=X_test.index, columns = range(len(all_mods.keys())))
        i = 0
        if weighted: # parms[0] is columns of selected features, parms[1] is weight of fold 
            if vote_type == 'hard':
                for mod, parms in all_mods.items():
                    vals = mod.predict(X_test[parms[0]])
                    preds['mod_%s'%i] = ((vals - 1) + vals) * parms[1]
                    i += 1
            else: # soft
                for mod, parms in all_mods.items():
                    vals = mod.predict_proba(X_test[parms[0]])[:,1]
                    preds['mod_%s'%i] = ((vals - 1) + vals) * parms[1]
                    i += 1

            pred_prob = preds.mean(axis=1)
            ambigs = [x for x in pred_prob.index if pred_prob[x] == 0]
            if len(ambigs)>0:
                print('there are %s ambiguous cases. Setting to hits...'%len(ambigs))
                print('ambiguous cases:',ambigs)
            pred_prob.loc[ambigs] = 0.01
            final_pred = pandas.Series(index = pred_prob.index)
            for x in pred_prob.index:
                if pred_prob[x] > 0:
                    final_pred[x] = 1
                else:
                    final_pred[x] = 0

        else: # unweighted
            if vote_type == 'hard':
                for mod, scols in all_mods.items():
                    preds['mod_%s'%i] = mod.predict(X_test[scols])
                    i += 1
            else: # soft
                for mod, scols in all_mods.items():
                    preds['mod_%s'%i] = mod.predict_proba(X_test[scols])[:,1]
                    i += 1
            
            pred_prob = preds.mean(axis=1)
            ambigs = [x for x in pred_prob.index if pred_prob[x] == 0.5]
            if len(ambigs)>0:
                print('there are %s ambiguous cases. Setting to hits...'%len(ambigs))
                print('ambiguous cases:',ambigs)
            pred_prob.loc[ambigs] = 0.51
            final_pred = pandas.Series(index = pred_prob.index)
            for x in pred_prob.index:
                if pred_prob[x] > 0.5:
                    final_pred[x] = 1
                else:
                    final_pred[x] = 0

    else: #regression
        cols = ['mod_%s'%x for x in range(len(all_mods))]
        preds = pandas.DataFrame(index=X_test.index, columns = range(len(all_mods.keys())))
        i = 0
        for mod, scols in all_mods.items():
            preds['mod_%s'%i] = mod.predict(X_test[scols])
            i += 1
        final_pred = preds.mean(axis=1)

    return final_pred

def balance_cohorts(df,col,ratio=2, save_df=False):
    train_ids,test_ids = [],[]
    subs = df.sort_values(col).index
    x = 0
    for i,sub in enumerate(subs):
        if i - x < ratio:
            train_ids.append(sub)
        else:
            test_ids.append(sub)
            x = i+1
    if save_df:
        tr_df = df.loc[train_ids]
        e_df = df.loc[test_ids]
        return tr_df, te_df
    else:
        return train_ids,test_ids

def feature_learning_optimizer(train, test, y, t_y, problem = 'regression', 
                               clfs = {'model': linear_model.LassoCV(cv=10)}, verbose = False,
                               ps = [None,0.2,0.1,0.05,0.01,0.005,0.001], folds = [2,3,5,10,20], 
                               scale = True, search = False, regcols = None, regdf = None, keep_cols = None,
                               outdir = None, cheat = False, optimize_on = 'acc', opt_key = None, output = 'light'):

    
    if type(optimize_on) == dict and opt_key == None:
        raise IOError('for the argument opt_key, please enter a string representing a key of optimize_one')

    ntests = len(clfs) * len(ps) * len(folds)
    print('running %s different tests'%(ntests))
    
    cols = ['clf','p','fold']
    if type(optimize_on) == dict:
        cols += list(optimize_on.keys())
    else:
        cols += ['acc']
    if problem == 'classification':
        cols += ['sens','spec']
    if cheat:
        cols += ['test_acc']
    results = pandas.DataFrame(index = range(ntests),columns = cols)
    
    i = 0
    
    if outdir:
        if not os.path.isdir(outdir):
            os.mkdir(outdir)
    
    for model,clf in clfs.items():
        print('*'*10, 'working on model',model,'*'*10)
        for p in ps:
            print('*'*5, 'p = ',str(p),'*'*5)
            for fold in folds:
                print('*'*2, 'using %s fold cross-validation'%fold,'*'*2)
                if type(optimize_on) == dict:
                    mod_output = kfold_feature_learning(train, test, y, t_y, clf, problem, fold, scale, verbose, 
                                                         search, p, regcols, regdf, keep_cols, output = 'light')
                    if len(mod_output[1]) == 0:
                        continue
                    for nm, met in optimize_on.items():
                        results.loc[results.index[i]][nm] = met(y,mod_output[1])
                    if cheat:
                        r,jnk = stats.pearsonr(t_y[test.index],mod_output[2])
                        results.loc[results.index[i]]['test_acc'] = r**2
                else:
                    val_res, t_res =  kfold_feature_learning(train, test, y, t_y, clf, problem, fold, scale, verbose, 
                                                         search, p, regcols, regdf, keep_cols, output = 'scores')
                    if problem == 'regression':
                        results.loc[results.index[i]]['acc'] = val_res
                        if cheat:
                            results.loc[results.index[i]]['test_acc'] = t_res
                    else:
                        results.loc[results.index[i]]['acc'] = val_res[-1]
                        results.loc[results.index[i]]['sens'] = val_res[0]
                        results.loc[results.index[i]]['spec'] = val_res[1]
                        if cheat:
                            results.loc[results.index[i]]['test_acc'] = t_res[-1]
                results.loc[results.index[i]]['clf'] = model
                results.loc[results.index[i]]['p'] = p
                results.loc[results.index[i]]['fold'] = fold
                
                if outdir:
                    if type(optimize_on) == str:
                        results = results.sort_values(optimize_on, axis=0, ascending = False)
                    else:
                        results = results.sort_values(opt_key, axis=0, ascending = False)
                    results.to_csv(os.path.join(outdir,'optimizer_results'))
                i += 1    
                    
    
    if type(optimize_on) == str:
        results = results.sort_values(optimize_on, axis=0, ascending = False)
    else:
        results = results.sort_values(opt_key, axis=0, ascending = False)
    results.index = range(len(results.index))
    
    fmod = results.ix[results.index[0],'clf']
    fp = results.ix[results.index[0],'p']
    ffold = results.ix[results.index[0],'fold']
    
    opt_model = 'model: %s \n p: %s \n fold %s '%(fmod, fp, ffold)
    
    print('optimal model is as follows \n', opt_model)
    print('maximum validation accuracy:', results.ix[results.index[0],optimize_on])
    
    
    print(('*'*10, 'RUNNING OPTIMAL MODEL','*'*10))
    fmodel_output = kfold_feature_learning(train, test, y, t_y, 
                                            clfs[fmod], problem, ffold, 
                                            scale, True, search, fp, regcols, regdf, 
                                            keep_cols, output = output)
    
    return fmodel_output
