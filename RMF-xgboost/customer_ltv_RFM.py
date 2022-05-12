"""
tutorial from: https://www.youtube.com/watch?v=s-32u6XdY7c
"""

import pandas as pd
import numpy as np
import joblib

import plydata.cat_tools as cat
import plotnine as pn

pn.options.dpi = 300

# read txt
df = pd.read_csv('dataset/CDNOW_master.txt',
    sep = '\s+',
    names = ['customer_id','date','quantity','price'],
    header = 0
)

# convert data to datatime
df = df.assign(
    date = lambda x:pd.to_datetime(x['date'].astype(str)) 
)


"""
COHORT analysis:

only the customer that has joined at the specific business day
"""
# set the range of initial purchase
df_first_purchase = df.sort_values(['customer_id','date'])\
    .groupby('customer_id')\
    .first()

earliest_purchase_date = df_first_purchase['date'].min()
latest_purchase_date = df_first_purchase['date'].max()
print('purchase date range: from {0} to {1}'.format(earliest_purchase_date,latest_purchase_date) )


# visualize purchase price within cohort by month
# parameter MS: month-start-frequency 
# more to see ref at : https://zhuanlan.zhihu.com/p/70353374
df.reset_index()\
    .set_index('date')\
    ['price']\
    .resample(rule = 'MS')\
    .sum().plot()

# visualize individual purchase behavior
account_id_list = df['customer_id'].unique()
id_selected = account_id_list[:10]

df_selected_id = df[df['customer_id'].isin(id_selected)]\
    .groupby(['customer_id','date'])\
    .sum()\
    .reset_index()

pn.ggplot(
    pn.aes('date','price',group = 'customer_id'),
    data = df_selected_id
)\
    +pn.geom_line()\
    +pn.geom_point()\
    +pn.facet_wrap('customer_id')\
    +pn.scale_x_date(
        breaks = "1 year",
        date_labels = '%Y'
    )
# from the graph right, we can see customer id = 1 and 2 only purchase once, customer 4 and 5 are lost


"""
MACHINE learning
1. How much will the customer purchase in the next 90 days? [Regression]
2. What's the probability a customer to make a purchase in the next 90 days?
"""

# time split
n_days = 90
max_date = df['date'].max()
cutoff = max_date - pd.to_timedelta(n_days, unit = 'd')

temporal_in_df = df[df['date'] <= cutoff ]
temporal_out_df = df[df['date'] > cutoff ]

# RFM feature engineering
targets_df = temporal_out_df.drop('quantity',axis = 1)\
    .groupby('customer_id')\
    .sum('price')\
    .rename({'price':'spend_90_total'},axis = 1)\
    .assign(spend_90_flag = 1) # spark->withcolumn; pandas->assign

# Recency (in temporal_in_df): the data difference from max date to the most recent purchase date
max_date_in = temporal_in_df['date'].max()
recency_feature_df = temporal_in_df\
    .groupby('customer_id')\
    .apply(
        lambda x: (x['date'].max() - max_date_in) / pd.to_timedelta(1,'day')
    )\
    .to_frame()\
    .set_axis(['recency'],axis = 1)

# frequency
feaquency_feature_df = temporal_in_df\
    .groupby('customer_id')\
    .size()\
    .reset_index(name='frequency')\
    .set_index('customer_id')

# monetary feature
monetary_feature_df = temporal_in_df\
    .groupby('customer_id')\
    .agg(
        {'price': ['mean','sum']}
    )\
    .set_axis(['price_mean','price_sum'],axis = 1)

# merge 3 features

rfm_features_df = pd.concat([recency_feature_df,feaquency_feature_df,monetary_feature_df],axis = 1)\
    .join(targets_df,how = 'left')\
    .fillna(0)


"""
MACHINE LEARNNING MODEL
"""
from xgboost import XGBClassifier,XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

x = rfm_features_df[['recency','frequency','price_mean','price_sum']]
# regressgion
y_spend = rfm_features_df[['spend_90_total']]
x_train,x_test,y_train_spend,y_test_spend = train_test_split(x,y_spend,test_size=0.2,random_state = 2022)

xgb_reg = XGBRegressor(
    objective= 'reg:squarederror',
    random_state = 2022
)
xgb_reg_model = GridSearchCV(
    estimator=xgb_reg,
    param_grid={'eta' :[0.05, 0.1,0.3],'max_depth': [3,5,7]},
    scoring = 'neg_mean_absolute_error',
    refit = True,
    cv = 5

)
   
xgb_reg_model.fit(x_train,y_train_spend)
xgb_reg_model.best_score_
xgb_reg_model.best_params_

yhat_reg = xgb_reg_model.predict(x_test)
test_error_reg = np.mean(abs(yhat_reg-y_test_spend.values))

# classification
y_prob = rfm_features_df[['spend_90_flag']]
x_train,x_test,y_train_prob,y_test_prob = train_test_split(x,y_prob,test_size=0.2,random_state = 2022)

xgb_clf = XGBClassifier(
    objective= 'binary:logistic',
    random_state = 2022
)

xgb_clf_model = GridSearchCV(
    estimator = xgb_clf,
    param_grid={'eta' :[0.01, 0.05,0.1],'max_depth': [2,4,6]},
    scoring = 'roc_auc',
    refit = True,
    cv = 5
)
xgb_clf_model.fit(x_train,y_train_prob)

xgb_clf_model.best_score_
xgb_clf_model.best_params_
xgb_clf_model.best_estimator_
y_prob_hat = xgb_clf_model.predict_proba(x_train)

# feature importance
# regression model
feature_importance_reg = xgb_reg_model\
    .best_estimator_\
    .get_booster()\
    .get_score(importance_type = 'gain')
feature_importance_reg_df = pd.DataFrame(
    data =  {
        'feature':list(feature_importance_reg.keys()),
        'value': list(feature_importance_reg.values())
    })\
    .assign(
        feature = lambda x: cat.cat_reorder(x['feature'],x['value'],ascending=True)
        )

pn.ggplot(
    pn.aes('feature', 'value'),
    data = feature_importance_reg_df
)\
    + pn.geom_col()\
    + pn.coord_flip()

# classification model
feature_importance_clf = xgb_clf_model\
    .best_estimator_\
    .get_booster()\
    .get_score(importance_type = 'gain')
feature_importance_clf_df = pd.DataFrame(
    data =  {
        'feature':list(feature_importance_clf.keys()),
        'value': list(feature_importance_clf.values())
    })\
    .assign(
        feature = lambda x: cat.cat_reorder(x['feature'],x['value'],ascending=True)
        )

pn.ggplot(
    pn.aes('feature', 'value'),
    data = feature_importance_clf_df
)\
    + pn.geom_col()\
    + pn.coord_flip()

"""
SAVE MODEL
"""
prediction_df


