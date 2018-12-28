# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np
import pandas as pd
from kaggle.competitions import twosigmanews

env = twosigmanews.make_env()
(marketdf, newsdf) = env.get_training_data()

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

import lightgbm as lgb


def prepare_data(marketdf, newsdf):
    # a bit of feature engineering
    marketdf['time'] = marketdf.time.dt.strftime("%Y%m%d").astype(int)
    marketdf['bartrend'] = marketdf['close'] / marketdf['open']
    marketdf['average'] = (marketdf['close'] + marketdf['open'])/2
    marketdf['pricevolume'] = marketdf['volume'] * marketdf['close']
    
    newsdf['time'] = newsdf.time.dt.strftime("%Y%m%d").astype(int)
    newsdf['assetCode'] = newsdf['assetCodes'].map(lambda x: list(eval(x))[0])
    newsdf['position'] = newsdf['firstMentionSentence'] / newsdf['sentenceCount']
    newsdf['coverage'] = newsdf['sentimentWordCount'] / newsdf['wordCount']

    # filter pre-2012 data, no particular reason
    marketdf = marketdf.loc[marketdf['time'] > 20120000]
    
    # get rid of extra junk from news data
    droplist = ['sourceTimestamp','firstCreated','sourceId','headline','takeSequence','provider','firstMentionSentence',
                'sentenceCount','bodySize','headlineTag','marketCommentary','subjects','audiences','sentimentClass',
                'assetName', 'assetCodes','urgency','wordCount','sentimentWordCount']
    newsdf.drop(droplist, axis=1, inplace=True)
    marketdf.drop(['assetName', 'volume'], axis=1, inplace=True)
    
    # combine multiple news reports for same assets on same day
    newsgp = newsdf.groupby(['time','assetCode'], sort=False).aggregate(np.mean).reset_index()
    
    # join news reports to market data, note many assets will have many days without news data
    return pd.merge(marketdf, newsgp, how='left', on=['time', 'assetCode'], copy=False) #, right_on=['time', 'assetCodes'])


def post_scaling(df):
    mean, std = np.mean(df), np.std(df)
    df = (df - mean)/ (std * 8)
    return np.clip(df,-1,1)



if __name__ == '__main__':
    print('============> preparing data...')
    cdf = prepare_data(marketdf, newsdf)    
    del marketdf, newsdf  # save the precious memory
    
    print('============> building training set...')
    targetcols = ['returnsOpenNextMktres10']
    traincols = [col for col in cdf.columns if col not in ['time', 'assetCode', 'universe'] + targetcols]
    
    dates = cdf['time'].unique()
    train = range(len(dates))[:int(0.93*len(dates))]
    val = range(len(dates))[int(0.93*len(dates)):]
    
    # we be classifyin
    cdf[targetcols[0]] = (cdf[targetcols[0]] > 0).astype(int)
    
    # train data
    Xt = cdf[traincols].fillna(0).loc[cdf['time'].isin(dates[train])].values
    Yt = cdf[targetcols].fillna(0).loc[cdf['time'].isin(dates[train])].values
    
    # validation data
    Xv = cdf[traincols].fillna(0).loc[cdf['time'].isin(dates[val])].values
    Yv = cdf[targetcols].fillna(0).loc[cdf['time'].isin(dates[val])].values
    
    print(Xt.shape, Xv.shape)
    
    
    print ('============> Training lightgbm')
    ########## params for lgb (binary version, 只能处理-1、1的分类)
    params = {"objective" : "binary",
              "metric" : "binary_logloss",
              "num_leaves" : 125, # originally 60
              "max_depth": -1,
              "learning_rate" : 0.0005,   # originally .01
              "bagging_fraction" : 0.9,  # subsample
              "feature_fraction" : 0.9,  # colsample_bytree
              "bagging_freq" : 5,        # subsample_freq
              "bagging_seed" : 2018,
              "verbosity" : -1 }
    
     
    # We can introduce other boosting algos, default is traditional gradient boosting decision tree
    # Other options include random forest (rf), dropouts meet multiple additive regression trees (dart), 
    # or gradient-based on one-side sampling 
    lgtrain, lgval = lgb.Dataset(Xt, Yt[:,0]), lgb.Dataset(Xv, Yv[:,0])
    lgbmodel = lgb.train(params, lgtrain, 2000, valid_sets=[lgtrain, lgval], early_stopping_rounds=300, verbose_eval=200)
    
    ########## 
    print("============> generating predictions...")
    preddays = env.get_prediction_days()
    th_ = 0
    for marketdf, newsdf, predtemplatedf in preddays:
        # print('------- |',th_,'day')
        cdf = prepare_data(marketdf, newsdf)
        Xp = cdf[traincols].fillna(0).values
        preds = lgbmodel.predict(Xp, num_iteration=lgbmodel.best_iteration) * 2 - 1
        predsdf = pd.DataFrame({'ast':cdf['assetCode'],'conf':post_scaling(preds)})
        predtemplatedf['confidenceValue'][predtemplatedf['assetCode'].isin(predsdf.ast)] = predsdf['conf'].values
        env.predict(predtemplatedf)
        th_ += 1 
    
    env.write_submission_file()
