#%%
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import numpy as np
import pandas as pd
from kaggle.competitions import twosigmanews

env = twosigmanews.make_env()
(marketdf, newsdf) = env.get_training_data()

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

import xgboost as xgb
from xgboost.sklearn import XGBRegressor
from xgboost import plot_importance



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




#%%
####################### params for lgb #######################
DATASET_SPLIT_METHOD = 'oneshot'   # 'oneshot' or 'kfold'
n_splits = 5 if DATASET_SPLIT_METHOD == 'kfold' else None
seed = 2018

#### (rmse method)
params = {  'silent': True,  # 设置成1则没有运行信息输出，最好是设置为0.是否在运行升级时打印消息。
            'learning_rate': 0.007,  # 如同学习率
            'min_child_weight': 1,  # 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
            ### ，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
            ### 这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
            'max_depth': 13,  # 构建树的深度，越大越容易过拟合
            'gamma': 0.1,  # 树的叶子节点上作进一步分区所需的最小损失减少,越大越保守，一般0.1、0.2这样子。
            # 'subsample':1,  # 随机采样训练样本 训练实例的子采样比
            'max_delta_step': 0,  # 最大增量步长，我们允许每个树的权重估计。
            # 'reg_alpha': 0.4,  # L1 正则项参数   #0.6
            'reg_lambda': 0.3,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
            # 'scale_pos_weight'=1, #如果取值大于0的话，在类别样本不平衡的情况下有助于快速收敛。平衡正负权重
            'objective': 'reg:linear',  # 多分类的问题 指定学习任务和相应的学习目标
            'n_estimators': 500,  # 树的个数 (800)
            'seed': seed,  # 随机种子
            'n_jobs': 16,  # cpu 线程数 默认最大
            'colsample_bytree':0.9,    #用来控制每棵随机采样的列数的占比(每一列是一个特征)
            'early_stopping_rounds':100, # 减弱过拟合的影响
            'eval_metric': 'rmse',      # 原来mlogloss
              }
print('============> have read params')


#%%
print('============> preparing data...')
cdf = prepare_data(marketdf, newsdf)    
del marketdf, newsdf  # save the precious memory


#%%
print('============> building training set...')
targetcols = ['returnsOpenNextMktres10']
traincols = [col for col in cdf.columns if col not in ['time', 'assetCode', 'universe'] + targetcols]


#%%
### we be classifyin
# cdf[targetcols[0]] = (cdf[targetcols[0]] > 0).astype(int)     # kaggle public kernel的数据截断方法
original_cdf_length = len(cdf.index)
left_threshold = -0.1
right_threshold = 0.1
trainset_drop_index_left = cdf[cdf[targetcols[0]] < left_threshold].index
cdf.drop(trainset_drop_index_left, axis='index', inplace=True)
cdf.reset_index(drop=True)
trainset_drop_index_right = cdf[cdf[targetcols[0]] > right_threshold].index
cdf.drop(trainset_drop_index_right, axis='index', inplace=True)
cdf.reset_index(drop=True)
print('------------ nows cdf/original cdf length =',len(cdf.index)/original_cdf_length)


#%%
#############################################################################################################
### 根据unique day来划分train-test集合
if DATASET_SPLIT_METHOD == 'oneshot':
    print ('============ using oneshot to split dataset ============')
    dates_unique = cdf['time'].unique()
    train = range(len(dates_unique))[:int(0.85*len(dates_unique))]
    val = range(len(dates_unique))[int(0.85*len(dates_unique)):]
    ### train data （by unique day)
    X_train = cdf[traincols].fillna(0).loc[cdf['time'].isin(dates_unique[train])].values
    y_train = cdf[targetcols].fillna(0).loc[cdf['time'].isin(dates_unique[train])].values
    ### validation data  （by unique day)
    X_valid = cdf[traincols].fillna(0).loc[cdf['time'].isin(dates_unique[val])].values
    y_valid = cdf[targetcols].fillna(0).loc[cdf['time'].isin(dates_unique[val])].values
    print(X_train.shape, X_valid.shape)
    ### begin xgboost training
    print ('============> Training xgboost')
    xgboost_model = XGBRegressor(**params)
    xgboost_model.fit(X_train, y_train, verbose=False)
    xx_pred = xgboost_model.predict(X_valid)
    valid_rmse = np.sqrt(mean_squared_error(y_true=y_valid, y_pred=xx_pred))
    print('------------ xgboost valid rmse:', valid_rmse)



#%%
#############################################################################################################
print("============> generating predictions...")
if DATASET_SPLIT_METHOD == 'oneshot':
    print ('============ using oneshot to predict testset ============')
    preddays = env.get_prediction_days()
    for marketdf, newsdf, predtemplatedf in preddays:
        cdf = prepare_data(marketdf, newsdf)
        Xp = cdf[traincols].fillna(0).values
        # preds = lgbmodel.predict(Xp, num_iteration=lgbmodel.best_iteration) * 2 - 1   # 原始的预测程序，对应的是0-1分类的情况
        preds = xgboost_model.predict(Xp)
        predsdf = pd.DataFrame({'ast':cdf['assetCode'],'conf':post_scaling(preds)})
        predtemplatedf['confidenceValue'][predtemplatedf['assetCode'].isin(predsdf.ast)] = predsdf['conf'].values
        env.predict(predtemplatedf)



### ouput final file    
#############################################################################################################
env.write_submission_file()


