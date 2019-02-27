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
TRAIN_DATA = "./data/train_dataset.csv"
TEST_DATA = "./data/test_dataset.csv"

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

    train['bigger_商场'] = train['近三个月月均商场出现次数'].map(lambda x: 1 if x >= 92 else 0)
    train = get_app_rate(train)
    train = get_abnormal_label(train,"当月网购类应用使用次数")
    train = get_abnormal_label(train, "当月物流快递类应用使用次数")
    train = get_abnormal_label(train, "当月金融理财类应用使用总次数")
    #train = get_abnormal_label(train, "当月视频播放类应用使用次数")
    #train = get_abnormal_label(train, "当月飞机类应用使用次数")
    train = get_abnormal_label(train, "当月火车类应用使用次数")
    train = get_abnormal_label(train, "当月旅游资讯类应用使用次数")
    train['当月网购类应用使用次数'] = np.log1p(train['当月网购类应用使用次数'])
    train['当月物流快递类应用使用次数'] = np.log1p(train['当月物流快递类应用使用次数'])
    train['当月金融理财类应用使用总次数'] = np.log1p(train['当月金融理财类应用使用总次数'])
    train['当月视频播放类应用使用次数'] = np.log1p(train['当月视频播放类应用使用次数'])
    train['当月飞机类应用使用次数'] = np.log1p(train['当月飞机类应用使用次数'])
    train['当月火车类应用使用次数'] = np.log1p(train['当月火车类应用使用次数'])
    train['当月旅游资讯类应用使用次数'] = np.log1p(train['当月旅游资讯类应用使用次数'])


    std = StandardScaler()
    minMax = MinMaxScaler()
    train['用户年龄归一化'] = std.fit_transform(train[['用户年龄']])
    train.drop("用户年龄", axis=1, inplace=True)
    train.drop("是否黑名单客户", axis=1, inplace=True)

    return train
if __name__ == "__main__":
    train = pd.read_csv(TRAIN_DATA, encoding="utf-8")
    test = pd.read_csv(TEST_DATA, encoding="utf-8")
    #print(train.columns)
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
    #print(train.describe())
    # print(train[['当月旅游资讯类应用使用次数','信用分']].groupby("当月旅游资讯类应用使用次数").mean())
    #print(train[['当月旅游资讯类应用使用次数', '信用分']].groupby("当月旅游资讯类应用使用次数").size())
    #print(train['当月旅游资讯类应用使用次数'].quantile(0.8))
    # from matplotlib import pyplot as plt
    # fig, ax = plt.subplots()
    # ax.scatter(x=train['当月火车类应用使用次数'], y=train['当月旅游资讯类应用使用次数'])
    # plt.ylabel('当月通话交往圈人数', fontsize=5)
    # plt.xlabel('信用分', fontsize=5)
    # plt.show()
    #sys.exit(-1)

    train = feature_processing(train)
    train.drop("用户编码", axis=1, inplace=True)
    #print(train.corr())
    test = feature_processing(test)


    model_train,model_test,model_label,model_score = train_test_split(train, train["信用分"], train_size=0.8, random_state=2019)
    model_train.drop("信用分", axis=1, inplace=True)
    model_test.drop("信用分", axis=1, inplace=True)
    test_data = test.drop("用户编码", axis=1)
    print(model_train.shape)
    print(model_test.shape)
    # l1正则化参数 reg_alpha
    # l2正则化参数 reg_lambda
    clf = lgb.LGBMRegressor(
        boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=2,
        max_depth=5, n_estimators=10000, objective='regression',
        subsample=0.6, colsample_bytree=0.6, subsample_freq=1,
        learning_rate=0.01, random_state=2018, n_jobs=-1, max_bin=250,
        min_child_weight=5,min_child_samples=10
    )
    is_model = False
    is_cv_seed = False
    if is_model:
        if is_cv_seed:
            y_test_seed = []
            y_train_seed = []
            y_test_score = []
            y_train_score = []
            seeds = range(1,1000,25)
            model_seeds = range(1,2000,200)
            for seed in seeds:
                model_train_seed, model_test_seed, model_label_seed, model_score_seed = train_test_split(model_train, model_label, train_size=0.9,random_state=seed)
                clf.fit(model_train_seed, model_label_seed, eval_set=[(model_train_seed, model_label_seed), (model_test_seed, model_score_seed)], early_stopping_rounds=200, verbose=-1,eval_metric="l2")
                #print(list(clf.feature_importances_))
                #print(model_train.columns)
                y_predict = clf.predict(model_test, num_iteration=clf.best_iteration_)
                print(clf.best_score_)
                y_test_seed.append(y_predict)
                y_train_score.append(clf.best_score_['training']['l2'])
                y_test_score.append(clf.best_score_['valid_1']['l2'])
                print(metrics_mae(y_predict.tolist(),list(model_score)))
                y_train = clf.predict(model_train, num_iteration=clf.best_iteration_)
                y_train_seed.append(y_train)
                print(metrics_mae(y_train.tolist(), list(model_label)))
                print(metrics_mae(y_train.tolist(), list(model_label)) - metrics_mae(y_predict.tolist(),list(model_score)))
            #计算权重
            y_train_w_sum = sum(y_train_score)
            y_test_w_sum = sum(y_test_score)
            y_train_w = np.array(minmax_adjust([y_train_w_sum/i for i in y_train_score])).reshape(len(seeds),1)
            y_test_w = np.array(minmax_adjust([y_test_w_sum/i for i in y_test_score])).reshape(len(seeds),1)
            print("权重总和：",y_train_w.sum())
            #y_narray = np.array(y_test_seed).mean(axis=0)
            #y_trainarray = np.array(y_train_seed).mean(axis=0)

            y_narray = (np.array(y_test_seed) * y_test_w).sum(axis=0)
            y_trainarray = (np.array(y_train_seed) * y_train_w).sum(axis=0)


            print("最终cv结果")
            print(metrics_mae(y_narray.tolist(), list(model_score)))
            print(metrics_mae(y_trainarray.tolist(), list(model_label)))
            print(metrics_mae(y_trainarray.tolist(), list(model_label)) - metrics_mae(y_narray.tolist(), list(model_score)))

        else:
            clf.fit(model_train, model_label,eval_set=[(model_train, model_label), (model_test, model_score)],early_stopping_rounds=200, verbose=-1, eval_metric="l2")
            y_predict = clf.predict(model_test, num_iteration=clf.best_iteration_)
            # for feature,feature_importance in zip(list(model_train.columns),list(clf.feature_importances_)):
            #     print(feature,feature_importance)
            print("test_score:",metrics_mae(y_predict.tolist(), list(model_score)))
            y_train = clf.predict(model_train, num_iteration=clf.best_iteration_)
            print("train_score",metrics_mae(y_train.tolist(), list(model_label)))
            print("train and test scala",metrics_mae(y_train.tolist(), list(model_label)) - metrics_mae(y_predict.tolist(), list(model_score)))

    else:
        label = train['信用分']
        train.drop("信用分", axis=1, inplace=True)
        y_test_seed = []
        y_train_seed = []
        y_test_score = []
        y_train_score = []
        seeds = range(1, 1000, 10)
        for seed in seeds:
            clf.fit(model_train, model_label, eval_set=[(model_train, model_label), (model_test, model_score)],early_stopping_rounds=200, verbose=-1, eval_metric="l2")
            #test['信用分'] = clf.predict(test_data)
            y_predict = clf.predict(test_data, num_iteration=clf.best_iteration_)
            y_test_seed.append(y_predict)
            y_train_score.append(clf.best_score_['training']['l2'])
            y_test_score.append(clf.best_score_['valid_1']['l2'])
            model_train, model_test, model_label, model_score = train_test_split(train, label, train_size=0.9,random_state=seed)
        y_train_w_sum = sum(y_train_score)
        y_test_w_sum = sum(y_test_score)
        y_train_w = np.array(minmax_adjust([y_train_w_sum / i for i in y_train_score])).reshape(len(seeds), 1)
        y_test_w = np.array(minmax_adjust([y_test_w_sum / i for i in y_test_score])).reshape(len(seeds), 1)
        print("权重总和：", y_train_w.sum())
        print("ok")
        #test['信用分'] = np.array(y_test_seed).mean(axis=0)
        test['信用分'] = (np.array(y_test_seed) * y_test_w).sum(axis=0)
        data_submit = pd.DataFrame()
        data_submit['id'] = test['用户编码']
        data_submit['score'] = test.apply(lambda row: int(round(row.信用分)), axis=1)
        data_submit.to_csv("submit/sc_lgb_0228_v1.csv", encoding="utf-8", index=False)




        #runage(25) 无权重
        # 0.06393453104021482
        # 0.06860486854449625
        # 0.004670337504281433
        #加权
        # 0.06393534857552043
        # 0.06862311157774827
        # 0.0046877630022278405


# l1 0.06393698370885655
# 0.0710482097627345
# 0.007111226053877956

# l2 0.06391287397020382
# 0.06932360954170161
# 0.005410735571497793

# test_score: 0.06385940712926422
# train_score 0.06879140382617788
# train and test scala 0.00493199669691366