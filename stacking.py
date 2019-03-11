import numpy as np
import pandas as pd
import time
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import RepeatedKFold
from utlis import *
from blending import Blending
class Stacking(object):
    def __init__(self, n_fold=10):
        self.n_fold = n_fold

    def get_stacking(self, oof_list, prediction_list, labels):
        train_stack = np.vstack(oof_list).transpose()
        test_stack = np.vstack(prediction_list).transpose()

        repeats = len(oof_list)
        #RepeatedKFold  p次k折交叉验证
        kfolder = RepeatedKFold(n_splits=self.n_fold, n_repeats=repeats, random_state=4590)
        kfold = kfolder.split(train_stack, labels)
        preds_list = list()
        stacking_oof = np.zeros(train_stack.shape[0])

        for train_index, vali_index in kfold:
            k_x_train = train_stack[train_index]
            k_y_train = labels.loc[train_index]
            k_x_vali = train_stack[vali_index]

            gbm = BayesianRidge(normalize=True)
            gbm.fit(k_x_train, k_y_train)

            k_pred = gbm.predict(k_x_vali)
            stacking_oof[vali_index] = k_pred

            preds = gbm.predict(test_stack)
            preds_list.append(preds)

        fold_mae_error = mean_absolute_error(labels, stacking_oof)
        print(f'stacking fold mae error is {fold_mae_error}')
        fold_score = 1 / (1 + fold_mae_error)
        print(f'fold score is {fold_score}')

        preds_columns = ['preds_{id}'.format(id=i) for i in range(self.n_fold * repeats)]
        preds_df = pd.DataFrame(data=preds_list)
        preds_df = preds_df.T
        preds_df.columns = preds_columns
        stacking_prediction = list(preds_df.mean(axis=1))

        return stacking_oof, stacking_prediction

if __name__ == "__main__":
    stacker = Stacking()
    oof_list = list()
    predict_list = list()

    v1_train = pd.read_csv('stacking/sc_lgb_train_0311_v2.csv', encoding='utf-8')
    v1_test = pd.read_csv('stacking/sc_lgb_test_0311_v2.csv', encoding='utf-8')
    oof_list.append(v1_train['predict_score'])
    predict_list.append(v1_test['score'])
    labels = v1_train['score']

    v4_train = pd.read_csv('stacking/sc_lgb_l1_train_0311_v2.csv', encoding='utf-8')
    v4_test = pd.read_csv('stacking/sc_lgb_l1_test_0311_v2.csv', encoding='utf-8')
    oof_list.append(v4_train['predict_score'])
    predict_list.append(v4_test['score'])


    v2_train = pd.read_csv('stacking/sc_xgb_train_0311_v1.csv', encoding='utf-8')
    v2_test = pd.read_csv('stacking/sc_xgb_test_0311_v1.csv', encoding='utf-8')
    oof_list.append(v2_train['predict_score'])
    predict_list.append(v2_test['score'])

    v3_train = pd.read_csv('stacking/sc_ctb_train_0307_v1.csv', encoding='utf-8')
    v3_test = pd.read_csv('stacking/sc_ctb_test_0307_v1.csv', encoding='utf-8')
    oof_list.append(v3_train['predict_score'])
    predict_list.append(v3_test['score'])

    train = pd.read_csv("./data/train_dataset.csv", encoding="utf-8")[['用户编码', '信用分']]
    train.columns = ['id', 'score']
    train_score_df = train
    test_score_df = v2_test[['id']]

    train_score_df["lgb"] = oof_list[0].tolist()
    test_score_df["lgb"] = predict_list[0].tolist()

    train_score_df["xgb"] = oof_list[1].tolist()
    test_score_df["xgb"] = predict_list[1].tolist()

    train_score_df["ctb"] = oof_list[2].tolist()
    test_score_df["ctb"] = predict_list[2].tolist()

    (stacking_oof, stacking_prediction) = stacker.get_stacking(oof_list, predict_list, labels)

    v2_test['score'] = stacking_prediction
    v2_test['score'] = v2_test['score'].apply(lambda item: int(round(item)))
    v2_test[['id','score']].to_csv("submit/sc_stacking_0312_v1.csv", index=False)


    import sys
    sys.exit(-1)

    # stacking
    mode_list = ['lgb', 'xgb', 'ctb']
    combinations_list = get_combinations(range(len(oof_list) + 1))
    for bin_item in combinations_list:
        oof = get_values_by_index(oof_list, bin_item)
        prediction = get_values_by_index(predict_list, bin_item)

        mode = get_values_by_index(mode_list, bin_item)
        mode.append('score')
        mode_name = '_'.join(mode)

        stacking_oof, stacking_prediction = Stacking().get_stacking(oof, prediction, train_score_df['score'])
        train_score_df[mode_name] = stacking_oof
        test_score_df[mode_name] = stacking_prediction

    # 加权分数
    get_ensemble_score(train_score_df)

    # blending
    best_weight = Blending(train_score_df).get_best_weight()
    score_array = get_score_array(test_score_df)
    test_score_df['score'] = get_blending_score(score_array, best_weight)
    test_score_df.to_csv('all_score.csv', index=False)

    sub_df = test_score_df[['id', 'score']]
    sub_df['score'] = sub_df['score'].apply(lambda item: int(round(item)))
    sub_df.to_csv('submittion.csv', index=False)






'''
stacking fold mae error is 14.666330839666823
fold score is 0.06383115550375208

stacking fold mae error is 14.6359780923003
fold score is 0.06395506530496066

stacking fold mae error is 14.63503905885699
fold score is 0.06395890641753892

stacking fold mae error is 14.623148940513563
fold score is 0.06400758283797864
'''

