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
from scipy import sparse
import inspect
from sklearn.metrics import mean_absolute_error
#from catboost import CatBoostRegressor

TRAIN_DATA = "./data/train_dataset.csv"
TEST_DATA = "./data/test_dataset.csv"
lgb_params_l1 = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'n_estimators': 10000,
    'metric': 'mae',
    'learning_rate': 0.01,
    'min_child_samples': 5,
    'min_child_weight': 0.01,
    'subsample_freq': 1,
    'num_leaves': 31,
    'max_depth': 5,
    'subsample': 0.6,
    'colsample_bytree': 0.6,
    'reg_alpha': 0,
    'reg_lambda': 5,
    'verbose': -1,
    'seed': 4590
}
lgb_params_l2 = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'n_estimators': 10000,
    'metric': 'mse',
    'learning_rate': 0.01,
    'min_child_samples': 5,
    'min_child_weight': 0.01,
    'subsample_freq': 1,
    'num_leaves': 31,
    'max_depth': 5,
    'subsample': 0.6,
    'colsample_bytree': 0.6,
    'reg_alpha': 0.15,
    'reg_lambda': 5,
    'verbose': -1,
    'seed': 2222
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
def log_feature(df):
    user_bill_features = ['缴费用户最近一次缴费金额（元）', '用户近6个月平均消费值（元）',
                          '用户账单当月总费用（元）', '用户当月账户余额（元）']
    log_features = ['当月网购类应用使用次数', '当月金融理财类应用使用总次数', '当月视频播放类应用使用次数']
    for col in user_bill_features + log_features:
        df[col] = df[col].map(lambda x: np.log1p(x))
    return df
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


def base_process(data):
    transform_value_feature = ['用户年龄', '用户网龄（月）', '当月通话交往圈人数', '近三个月月均商场出现次数', '当月网购类应用使用次数', '当月物流快递类应用使用次数'
        , '当月金融理财类应用使用总次数', '当月视频播放类应用使用次数', '当月飞机类应用使用次数', '当月火车类应用使用次数', '当月旅游资讯类应用使用次数']
    user_fea = ['缴费用户最近一次缴费金额（元）', '用户近6个月平均消费值（元）', '用户账单当月总费用（元）', '用户当月账户余额（元）']
    log_features = ['当月网购类应用使用次数', '当月金融理财类应用使用总次数', '当月物流快递类应用使用次数', '当月视频播放类应用使用次数']

    # 处理离散点
    for col in transform_value_feature + user_fea + log_features:
        # 取出最高99.9%值
        ulimit = np.percentile(data[col].values, 99)
        # 取出最低0.1%值
        llimit = np.percentile(data[col].values, 0.1)
        data.loc[data[col] > ulimit, col] = ulimit
        data.loc[data[col] < llimit, col] = llimit

    for col in user_fea + log_features:
        data[col] = data[col].map(lambda x: np.log1p(x))

    return data

def map_discretize(x):
    if x == 0:
        return 0
    elif x <= 5:
        return 1
    elif x <= 15:
        return 2
    elif x <= 50:
        return 3
    elif x <= 100:
        return 4
    else:
        return 5
def feature_processing(train):
    train['up_zero'] = train['缴费用户最近一次缴费金额（元）'].map(lambda x: 0 if x != 0 else 1)
    train['avg_zero'] = train['用户近6个月平均消费值（元）'].map(lambda x: 0 if x != 0 else 1)
    train['all_zero'] = train['用户账单当月总费用（元）'].map(lambda x: 0 if x != 0 else 1)
    train['zero_age'] = train['用户年龄'].map(lambda x: 0 if x != 0 else 1)
    train['top_up_amount_offline'] = 0
    train['top_up_amount_offline'][
        (train['缴费用户最近一次缴费金额（元）'] % 10 == 0) & train['缴费用户最近一次缴费金额（元）'] != 0] = 1
    train['current_fee_stability'] = train['用户账单当月总费用（元）'] / (train['用户近6个月平均消费值（元）'] + 1)
    train['use_left_rate'] = train['用户账单当月总费用（元）'] / (train['用户当月账户余额（元）'] + 1)
    train['交通类应用使用次数'] = train['当月飞机类应用使用次数'] + train['当月火车类应用使用次数']
    #尝试特征
    train['是否去过高档商场'] = train['当月是否到过福州山姆会员店'] + train['当月是否逛过福州仓山万达']
    train['是否去过高档商场'] = train['是否去过高档商场'].map(lambda x: 1 if x >= 1 else 0)
    train['是否_商场_电影'] = train['是否去过高档商场'] * train['当月是否看电影']
    train['是否_商场_旅游_体育馆'] = train['是否去过高档商场'] * train['当月是否景点游览'] * train['当月是否体育场馆消费']
    #app使用次数
    discretize_features = ['交通类应用使用次数', '当月物流快递类应用使用次数', '当月飞机类应用使用次数', '当月火车类应用使用次数', '当月旅游资讯类应用使用次数']
    train['6个月占比总费用'] = train['用户近6个月平均消费值（元）'] / train['用户账单当月总费用（元）'] + 1
    train["用户年龄"] = train["用户年龄"].replace(0, train["用户年龄"].mode())
    train = base_process(train) # 对所有特征的异常值进行处理
    for col in discretize_features[:]:
        train[col] = train[col].map(lambda x: map_discretize(x))
    #####################################################3

    train['当月账单是否超过平均消费额'] = train['用户账单当月总费用（元）'] - train['用户近6个月平均消费值（元）']
    train['缴费金额是否能覆盖当月账单'] = train['缴费用户最近一次缴费金额（元）'] - train['用户账单当月总费用（元）']
    train['最近一次缴费是否超过平均消费额'] = train['缴费用户最近一次缴费金额（元）'] - train['用户近6个月平均消费值（元）']


    '''
    fold score is 0.06382183981577505
    '''



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


    train_labels = train["信用分"]
    train.drop("用户编码", axis=1, inplace=True)
    train.drop("信用分", axis=1, inplace=True)
    test_data = test.drop("用户编码", axis=1)

    # test_data = feature_processing(test_data)
    # train = feature_processing(train)
    train_shape = train.shape[0]
    test_shape = test.shape[0]
    data = pd.concat([train, test_data], ignore_index=True)
    data = feature_processing(data)
    train = data[0:train_shape]
    test_data = data[train_shape:]

    train_x = sparse.csr_matrix(train.values)
    test_x = sparse.csr_matrix(test_data.values)

    n_fold = 10
    seed = 22
    kfolder = KFold(n_splits=n_fold, shuffle=True, random_state=seed)
    kfold = kfolder.split(train_x, train_labels)
    preds_list = list()
    oof = np.zeros(train.shape[0])
    count_fold = 0
    seeds = range(1, 3)
    for train_index, vali_index in kfold:
        print(count_fold)
        count_fold = count_fold + 1
        for sed in seeds:

            k_x_train = train_x[train_index]
            k_y_train = train_labels.loc[train_index]
            k_x_vali = train_x[vali_index]
            k_y_vali = train_labels.loc[vali_index]

            lgb_params_l1['seed'] = sed
            lgb_model = lgb.LGBMRegressor(**lgb_params_l1)
            lgb_model = lgb_model.fit(k_x_train, k_y_train, eval_set=[(k_x_train, k_y_train), (k_x_vali, k_y_vali)],
                          early_stopping_rounds=200, verbose=False,eval_metric="l1")
            iteration_kwargs = get_iteration_kwargs(lgb_model)
            k_pred = lgb_model.predict(k_x_vali, **iteration_kwargs)
            oof[vali_index] = k_pred

            preds = lgb_model.predict(test_x, **iteration_kwargs)
            preds_list.append(preds)

    fold_mae_error = mean_absolute_error(train_labels, oof)
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

        # sub_df.to_csv('stacking/sc_lgb_test_l2.csv', index=False)
        # train['predict_score'] = oof
        # train['score'] = train_labels
        # train[['score','predict_score']].to_csv('stacking/sc_lgb_train_l2.csv', index=False)

        sub_df.to_csv('stacking/sc_lgb_l1_test.csv', index=False)
        train['predict_score'] = oof
        train['score'] = train_labels
        train[['score','predict_score']].to_csv('stacking/sc_lgb_l1_train.csv', index=False)



'''
l2

 fold mae error is 14.658279639001737
fold score is 0.0638639763150732


l1

fold score is 0.06387295110332751
'''