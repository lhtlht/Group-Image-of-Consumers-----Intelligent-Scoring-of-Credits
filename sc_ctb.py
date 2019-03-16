import pandas as pd
import numpy as np
import os
import sys
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from scipy import sparse
import inspect
from sklearn.metrics import mean_absolute_error
#from catboost import CatBoostRegressor

TRAIN_DATA = "./data/train_dataset.csv"
TEST_DATA = "./data/test_dataset.csv"
ctb_params = {
    'n_estimators': 10000,
    'learning_rate': 0.01,
    'random_seed': 4590,
    'reg_lambda': 0.08,
    'subsample': 0.7,
    'bootstrap_type': 'Bernoulli',
    'boosting_type': 'Plain',
    'one_hot_max_size': 10,
    'rsm': 0.5,
    'leaf_estimation_iterations': 5,
    'use_best_model': True,
    'max_depth': 6,
    'verbose': -1,
    'thread_count': 4
}
lgb_params_good = {
    'boosting_type': 'gbdt',
    'objective': 'mae',
    'n_estimators': 10000,
    'metric': 'mae',
    'learning_rate': 0.01,
    'min_child_samples': 5,
    'min_child_weight': 0.01,
    'subsample_freq': 1,
    'num_leaves': 40,
    'max_depth': 7,
    'subsample': 0.42,
    'colsample_bytree': 0.48,
    'reg_alpha': 0.15,
    'reg_lambda': 5,
    'verbose': -1,
    'seed': 4590
}
def metrics_mae(label, score):
    sum = 0.0
    for (l, s) in zip(label, score):
        sum += abs(int(round(l))-s)
    return 1.0/(1+sum/len(label))

def minmax_adjust(m_list):
    m_sum = sum(m_list)
    return [i/m_sum for i in m_list]

def get_abnormal_label(train, name):
    train[name + "_1"] = train.apply(lambda row: 1 if row[name] ==0 else 0, axis=1)
    train[name + "_2"] = train.apply(lambda row: 1 if row[name] > 0 and row[name] <= 10 else 0, axis=1)
    train[name + "_3"] = train.apply(lambda row: 1 if row[name] > 10 and row[name] <= 100 else 0, axis=1)
    train[name + "_4"] = train.apply(lambda row: 1 if row[name] > 100 and row[name] <= 1000 else 0, axis=1)
    train[name + "_4"] = train.apply(lambda row: 1 if row[name] > 1000 else 0, axis=1)
    return train

def get_app_rate(dataset):
    dataset = dataset.copy()

    app_num_columns = ['当月网购类应用使用次数', '当月物流快递类应用使用次数', '当月金融理财类应用使用总次数', '当月视频播放类应用使用次数', '当月飞机类应用使用次数',
                       '当月火车类应用使用次数', '当月旅游资讯类应用使用次数']
    dataset['helper_sum'] = dataset[app_num_columns].apply(lambda item: np.log1p(np.sum(item)), axis=1)

    for column in app_num_columns:
        column_name = f'{column}_rate'
        dataset[column_name] = np.log1p(dataset[column]) / dataset['helper_sum']

    #dataset = dataset.drop(columns=['helper_sum'])
    return dataset

def feature_processing(train):
    train['用户当月消费是否低于近6个月'] = train.apply(lambda row: 1 if row['用户账单当月总费用（元）'] > row['用户近6个月平均消费值（元）'] else 0, axis=1)
    train['用户余额能支持月数'] = train.apply(lambda row: row['用户当月账户余额（元）'] / (row['用户近6个月平均消费值（元）'] + 1.0), axis=1)
    train['用户最近缴费支持月数'] = train.apply(lambda row: (row['缴费用户最近一次缴费金额（元）']) / (row['用户近6个月平均消费值（元）'] + 1.0), axis=1)
    train['up_zero'] = train['缴费用户最近一次缴费金额（元）'].map(lambda x: 0 if x != 0 else 1)
    train['avg_zero'] = train['用户近6个月平均消费值（元）'].map(lambda x: 0 if x != 0 else 1)
    train['all_zero'] = train['用户账单当月总费用（元）'].map(lambda x: 0 if x != 0 else 1)
    train['zero_age'] = train['用户年龄'].map(lambda x: 0 if x != 0 else 1)
    train['充值金额是否整数'] = 0
    train['充值金额是否整数'][(train['缴费用户最近一次缴费金额（元）'] % 10 == 0) & train['缴费用户最近一次缴费金额（元）'] != 0] = 1
    train['当月花费的稳定性'] = train['用户账单当月总费用（元）'] / (train['用户近6个月平均消费值（元）'] + 1)
    train['use_left_rate'] = train['用户账单当月总费用（元）'] / (train['用户当月账户余额（元）'] + 1)

    train['是否去过高档商场'] = train['当月是否逛过福州仓山万达'] * train['当月是否到过福州山姆会员店']
    train['交通类应用使用次数'] = train['当月飞机类应用使用次数'] + train['当月火车类应用使用次数']
    train["用户年龄"] = train["用户年龄"].replace(0, train["用户年龄"].mode())
    train['当月账单是否超过平均消费额'] = train['用户账单当月总费用（元）'] - train['用户近6个月平均消费值（元）']
    train['缴费金额是否能覆盖当月账单'] = train['缴费用户最近一次缴费金额（元）'] - train['用户账单当月总费用（元）']
    train['最近一次缴费是否超过平均消费额'] = train['缴费用户最近一次缴费金额（元）'] - train['用户近6个月平均消费值（元）']


    std = StandardScaler()
    minMax = MinMaxScaler()
    train['用户年龄归一化'] = std.fit_transform(train[['用户年龄']])
    train.drop("用户年龄", axis=1, inplace=True)

    return train

def get_iteration_kwargs(gbm):
    predict_args = inspect.getfullargspec(gbm.predict).args
    if hasattr(gbm, 'best_iteration_'):
        best_iteration = getattr(gbm, 'best_iteration_')
        if 'num_iteration' in predict_args:
            iteration_kwargs = {'num_iteration': best_iteration}
        elif 'ntree_end' in predict_args:
            iteration_kwargs = {'ntree_end': best_iteration}
        else:
            raise ValueError()
    elif hasattr(gbm, 'best_ntree_limit'):
        best_iteration = getattr(gbm, 'best_ntree_limit')
        if 'ntree_limit' in predict_args:
            iteration_kwargs = {'ntree_limit': best_iteration}
        else:
            raise ValueError()
    else:
        raise ValueError()
    return iteration_kwargs
if __name__ == "__main__":
    train = pd.read_csv(TRAIN_DATA, encoding="utf-8")
    test = pd.read_csv(TEST_DATA, encoding="utf-8")
    '''
    Index(['用户编码', '用户实名制是否通过核实', '用户年龄', '是否大学生客户', '是否黑名单客户', '是否4G不健康客户',
       '用户网龄（月）', '用户最近一次缴费距今时长（月）', '缴费用户最近一次缴费金额（元）', '用户近6个月平均消费值（元）',
       '用户账单当月总费用（元）', '用户当月账户余额（元）', '缴费用户当前是否欠费缴费', '用户话费敏感度', '当月通话交往圈人数',
       '是否经常逛商场的人', '近三个月月均商场出现次数', '当月是否逛过福州仓山万达', '当月是否到过福州山姆会员店', '当月是否看电影',
       '当月是否景点游览', '当月是否体育场馆消费', '当月网购类应用使用次数', '当月物流快递类应用使用次数',
       '当月金融理财类应用使用总次数', '当月视频播放类应用使用次数', '当月飞机类应用使用次数', '当月火车类应用使用次数',
       '当月旅游资讯类应用使用次数', '信用分'],
      dtype='object')
    '''

    train = feature_processing(train)
    train_labels = train["信用分"]
    train.drop("用户编码", axis=1, inplace=True)
    train.drop("信用分", axis=1, inplace=True)
    test = feature_processing(test)
    test_data = test.drop("用户编码", axis=1)

    # train_x = sparse.csr_matrix(train.values)
    # test_x = sparse.csr_matrix(test_data.values)
    train_x = train
    test_x = test_data

    n_fold = 10
    seed = 22
    kfolder = KFold(n_splits=n_fold, shuffle=True, random_state=seed)
    kfold = kfolder.split(train_x, train_labels)
    preds_list = list()
    oof = np.zeros(train.shape[0])
    count_fold = 0
    seeds = range(1, 1000, 500)
    for train_index, vali_index in kfold:
        print(count_fold)
        count_fold = count_fold + 1
        for sed in seeds:
            ctb_params['random_seed'] = sed
            k_x_train = train_x.loc[train_index]
            k_y_train = train_labels.loc[train_index]
            k_x_vali = train_x.loc[vali_index]
            k_y_vali = train_labels.loc[vali_index]

            ctb_model = CatBoostRegressor(**ctb_params)
            ctb_model = ctb_model.fit(k_x_train, k_y_train, eval_set=[(k_x_train, k_y_train), (k_x_vali, k_y_vali)],
                          early_stopping_rounds=200, verbose=False)
            iteration_kwargs = get_iteration_kwargs(ctb_model)
            k_pred = ctb_model.predict(k_x_vali, **iteration_kwargs)
            oof[vali_index] = k_pred

            preds = ctb_model.predict(test_x, **iteration_kwargs)
            preds_list.append(preds)

    fold_mae_error = mean_absolute_error(train_labels, oof)
    print(' fold mae error is',fold_mae_error)
    fold_score = 1 / (1 + fold_mae_error)
    print(f'fold score is {fold_score}')

    preds_columns = ['preds_{id}'.format(id=i) for i in range(n_fold*len(seeds))]
    preds_df = pd.DataFrame(data=preds_list)
    preds_df = preds_df.T
    preds_df.columns = preds_columns
    preds_list = list(preds_df.mean(axis=1))
    prediction = preds_list
    is_save = True
    if is_save:
        sub_df = pd.DataFrame({'id': test['用户编码'],
                               'score': prediction})
        #sub_df['score'] = sub_df['score'].apply(lambda item: int(round(item)))
        #sub_df.to_csv('submit/sc_lgb_0301_v2.csv', index=False)
        sub_df.to_csv('stacking/sc_ctb_test.csv', index=False)
        train['predict_score'] = oof
        train['score'] = train_labels
        train[['score','predict_score']].to_csv('stacking/sc_ctb_train.csv', index=False)

#  fold mae error is 14.675924985134769
# fold score is 0.0637920888845975

'''
stacking fold mae error is 14.625699581822275
fold score is 0.06399713464114735

 fold mae error is 14.656107345270174
fold score is 0.06387283747783624
'''