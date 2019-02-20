import pandas as pd
import os
import sys
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
TRAIN_DATA = "./data/train_dataset.csv"
TEST_DATA = "./data/test_dataset.csv"

def metrics_mae(label, score):
    sum = 0.0
    for (l, s) in zip(label, score):
        sum += abs(int(round(l))-s)
    return 1.0/(1+sum/len(label))

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
    #print(train[['用户实名制是否通过核实']].groupby("用户实名制是否通过核实").size())
    #print(train[['当月通话交往圈人数','信用分']].groupby("当月通话交往圈人数").mean())
    #train.apply(lambda row: int(round(row.信用分)), axis=1)
    # print(train.isnull().any())
    # from matplotlib import pyplot as plt
    # fig, ax = plt.subplots()
    # ax.scatter(x=train['当月通话交往圈人数'], y=train['信用分'])
    # plt.ylabel('当月通话交往圈人数', fontsize=5)
    # plt.xlabel('信用分', fontsize=5)
    # plt.show()

    # col_list = list(train.columns)
    # for col in col_list:
    #     print(col)
    #     if (col=="用户编码"):
    #         continue
    #     print(train[col].skew())
    # sys.exit(-1)


    train['用户当月消费是否低于近6个月'] = train.apply(lambda row: 1 if row['用户账单当月总费用（元）'] > row['用户近6个月平均消费值（元）'] else 0, axis=1)
    train['用户余额能支持月数'] = train.apply(lambda row: row['用户当月账户余额（元）'] / (row['用户近6个月平均消费值（元）']+1.0) , axis=1)
    train['用户最近缴费支持月数'] = train.apply(lambda row: (row['缴费用户最近一次缴费金额（元）']) / (row['用户近6个月平均消费值（元）']+1.0), axis=1)
    train['up_zero'] = train['缴费用户最近一次缴费金额（元）'].map(lambda x: 0 if x != 0 else 1)
    train['avg_zero'] = train['用户近6个月平均消费值（元）'].map(lambda x: 0 if x != 0 else 1)
    train['all_zero'] = train['用户账单当月总费用（元）'].map(lambda x: 0 if x != 0 else 1)
    train['zero_age'] = train['用户年龄'].map(lambda x: 0 if x != 0 else 1)
    train['top_up_amount_offline'] = 0
    train['top_up_amount_offline'][
        (train['缴费用户最近一次缴费金额（元）'] % 10 == 0) & train['缴费用户最近一次缴费金额（元）'] != 0] = 1
    train['current_fee_stability'] = train['用户账单当月总费用（元）'] / (train['用户近6个月平均消费值（元）'] + 1)
    train['use_left_rate'] = train['用户账单当月总费用（元）'] / (train['用户当月账户余额（元）'] + 1)
    std = StandardScaler()
    minMax = MinMaxScaler()
    #stacking_y = pd.read_csv('xgb_stacking_y.csv', encoding="utf-8")
    #train['stacking_y'] = minMax.fit_transform(stacking_y[['xgb_stacking_y']])
    train['用户年龄归一化'] = std.fit_transform(train[['用户年龄']])
    train.drop("用户年龄", axis=1, inplace=True)
    train.drop("是否黑名单客户", axis=1, inplace=True)
    train.drop("用户编码", axis=1, inplace=True)

    test['用户当月消费是否低于近6个月'] = test.apply(lambda row: 1 if row['用户账单当月总费用（元）'] > row['用户近6个月平均消费值（元）'] else 0, axis=1)
    test['用户余额能支持月数'] = test.apply(lambda row: row['用户当月账户余额（元）'] / (row['用户近6个月平均消费值（元）'] + 1.0), axis=1)
    test['用户最近缴费支持月数'] = test.apply(lambda row: (row['缴费用户最近一次缴费金额（元）']) / (row['用户近6个月平均消费值（元）'] + 1.0), axis=1)
    test['up_zero'] = test['缴费用户最近一次缴费金额（元）'].map(lambda x: 0 if x != 0 else 1)
    test['avg_zero'] = test['用户近6个月平均消费值（元）'].map(lambda x: 0 if x != 0 else 1)
    test['all_zero'] = test['用户账单当月总费用（元）'].map(lambda x: 0 if x != 0 else 1)
    test['zero_age'] = test['用户年龄'].map(lambda x: 0 if x != 0 else 1)
    test['top_up_amount_offline'] = 0
    test['top_up_amount_offline'][
        (test['缴费用户最近一次缴费金额（元）'] % 10 == 0) & test['缴费用户最近一次缴费金额（元）'] != 0] = 1
    test['current_fee_stability'] = test['用户账单当月总费用（元）'] / (test['用户近6个月平均消费值（元）'] + 1)
    test['use_left_rate'] = test['用户账单当月总费用（元）'] / (test['用户当月账户余额（元）'] + 1)


    test['用户年龄归一化'] = std.fit_transform(test[['用户年龄']])
    test.drop("用户年龄", axis=1, inplace=True)

    model_train,model_test,model_label,model_score = train_test_split(train, train["信用分"], train_size=0.8, random_state=2019)
    model_train.drop("信用分", axis=1, inplace=True)
    model_test.drop("信用分", axis=1, inplace=True)
    test_data = test.drop("用户编码", axis=1)
    print(model_train.shape)
    print(model_test.shape)

    clf = lgb.LGBMRegressor(
        boosting_type='gbdt', num_leaves=63, reg_alpha=0.0, reg_lambda=1,
        max_depth=5, n_estimators=10000, objective='regression',
        subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
        learning_rate=0.01, min_child_weight=50, random_state=2018, n_jobs=-1, max_bin=250
    )
    is_model = True
    if is_model:
        clf.fit(model_train, model_label, eval_set=[(model_train, model_label), (model_test, model_score)], early_stopping_rounds=200, verbose=10)
        print(list(clf.feature_importances_))
        print(model_train.columns)
        y_predict = clf.predict(model_test, num_iteration=clf.best_iteration_)
        print(metrics_mae(y_predict.tolist(),list(model_score)))
    else:
        label = train['信用分']
        train.drop("信用分", axis=1, inplace=True)
        clf.fit(model_train, model_label, eval_set=[(model_train, model_label), (model_test, model_score)],early_stopping_rounds=200, verbose=10)
        #clf.fit(train, label)
        test['信用分'] = clf.predict(test_data)

        data_submit = pd.DataFrame()
        data_submit['id'] = test['用户编码']
        data_submit['score'] = test.apply(lambda row: int(round(row.信用分)), axis=1)
        data_submit.to_csv("submit/sc_lgb_0221_v2.csv", encoding="utf-8", index=False)

#baseline : 0.06304295747122089
#0.0632519070449974
#0.06329474463735277
#0.06368088236230601
#0.06373201960396924
#0.06375721253466798
#0.06376575014028465
#0.06377103647065575
#0.063817838362179
#0.06383820844451821
#0.06385410613829522



