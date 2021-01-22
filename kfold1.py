from xgboost import XGBRegressor
import eli5
from eli5.sklearn import PermutationImportance
from sklearn.model_selection import KFold

xgbo = XGBRegressor(base_score=None, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=0.7, gamma=0.1, learning_rate=0.1, max_delta_step=0,
       max_depth=9, min_child_weight=1, missing=None, n_estimators=450,
       n_jobs=-1, nthread=None, objective='reg:squarederror', random_state=0,
       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,eta=0.01,
       silent=True, subsample=1)



EXCEPT = ['ITEM_ID',
          'ITEM_ID_Click',
          'PHOTO_NUM',
          'HALL',
          'TOILET',
          'NEAR_SHOP','NEAR_PARK','NEAR_HOSPITAL','NEAR_MARKET','NEAR_SUPER','NEAR_NIGHT_MARKET','NEAR_BUS',
          'NEAR_TRAIN',
          'ENCODE_kind',
          'traffic','life_asset',
          'ENCODE_fitment',
          'ENCODE_toward','ENCODE_shape','ENCODE_parking_type',
          'ROOM','BALCONY',
          'MONTH_PAY',
          'AREA',
          'diff_TOILET',
          'diff_HALL',
          'diff_BALCONY',
          'diff_PUBLIC_RATE',
          'diff_FEE',
          'diff_ROOM',
          'diff_AREA',
          'PUBLIC_RATE',
          'NEAR_MRT','NEAR_SCHOOL', #進學校 近捷運 反而沒有加分?
          'PRICE',
          'UNIT_PRICE',
          'AGE','FEE',
          'diff_AGE','diff_UNIT_PRICE'
         ] # +['diff_'+col for col in COLS1]
EXCEPT = list(set(dft.columns).intersection(EXCEPT))
dfsp = dft.sample(60000)
X = dfsp.drop(['label']+EXCEPT,axis=1)
y = dfsp.label

print("訓練數量: %s" % len(X))

avgScore = 0
n_splits = 5
kf = KFold(n_splits=n_splits,shuffle=True)
for train_index, test_index in kf.split(X):
#     print(len(train_index)/len(test_index))
    X_train = X.iloc[train_index]
    y_train = y.iloc[train_index]
    X_test = X.iloc[test_index]
    y_test = y.iloc[test_index]
    xgbo.fit(X_train, y_train)
    
    score = float(xgbo.score(X_train, y_train)) 
    print("Train分數:",score)
    score = float(xgbo.score(X_test, y_test))
    print("Test分數:",score)
    avgScore += score
    
perm = PermutationImportance(xgbo, random_state=1).fit(X_test, y_test)
display(eli5.show_weights(perm, feature_names = X_test.columns.tolist(),top=100))
    
print("平均分數:",avgScore/n_splits)
