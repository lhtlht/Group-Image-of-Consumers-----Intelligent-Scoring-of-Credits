import pandas as pd
import os
import sys
from sklearn.model_selection import train_test_split
import lightgbm as lgb

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
    print(train.columns)
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
    print(train[['当月飞机类应用使用次数','信用分']].groupby("当月飞机类应用使用次数").mean())
    #train.apply(lambda row: int(round(row.信用分)), axis=1)
    #sys.exit(-1)
    #train['is_plane_user'] = train.apply(lambda row: 1 if row.当月飞机类应用使用次数>row.当月火车类应用使用次数 else 0, axis=1)
    #train['out_counts'] = train.apply(lambda row: row.当月飞机类应用使用次数 + row.当月火车类应用使用次数, axis=1)
    #train['shop_counts'] = train.apply(lambda row: row.当月网购类应用使用次数 + row.当月物流快递类应用使用次数, axis=1)
    #train['用户当月余额总数'] = train.apply(lambda row: row['用户账单当月总费用（元）'] + row['用户当月账户余额（元）'], axis=1)
    #train['用户开网年龄'] = train.apply(lambda row: row['用户年龄'] - row['用户网龄（月）']/12.0, axis=1)

    train.drop("用户编码", axis=1, inplace=True)
    #train.drop("是否大学生客户", axis=1, inplace=True)
    model_train,model_test,model_label,model_score = train_test_split(train, train["信用分"], train_size=0.8)
    model_train.drop("信用分", axis=1, inplace=True)
    model_test.drop("信用分", axis=1, inplace=True)
    # model_train.drop("是否大学生客户", axis=1, inplace=True)
    # model_test.drop("是否大学生客户", axis=1, inplace=True)
    test_data = test.drop("用户编码", axis=1)
    print(model_train.shape)
    print(model_test.shape)

    clf = lgb.LGBMRegressor(
        boosting_type='gbdt', num_leaves=125, reg_alpha=0.0, reg_lambda=1,
        max_depth=7, n_estimators=3000, objective='regression',
        subsample=0.9, colsample_bytree=0.7, subsample_freq=1,
        learning_rate=0.01, min_child_weight=50, random_state=2018, n_jobs=-1, max_bin=200
    )
    is_model = True
    if is_model:
        clf.fit(model_train, model_label)
        print(list(clf.feature_importances_))
        print(model_train.columns)
        y_predict = clf.predict(model_test)
        print(metrics_mae(y_predict.tolist(),list(model_score)))
    else:
        label = train['信用分']
        train.drop("信用分", axis=1, inplace=True)
        train.drop("用户编码", axis=1, inplace=True)
        clf.fit(train, label)
        test['信用分'] = clf.predict(test_data)

        data_submit = pd.DataFrame()
        data_submit['id'] = test['用户编码']
        data_submit['score'] = test.apply(lambda row: int(round(row.信用分)), axis=1)
        data_submit.to_csv("sc_0214_v2.csv", encoding="utf-8", index=False)


#0.06377062979873989


