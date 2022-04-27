a={'target': 0.884486305802127, 'params': {'colsample_bytree': 0.8851239118822929, 'gamma': 4.099490165857144, 'learning_rate': 0.11180160668743563, 'max_depth': 2.031832257759685, 'min_child_weight': 0.09953018751866605, 'n_estimators': 198.71845955006978, 'reg_alpha': 1.1996184004896795, 'reg_lambda': 0.5585333812992521}}

# a['params']['max_depth']=int(a['params']['max_depth'])
# a['params']['n_estimators']=int(a['params']['n_estimators'])


a['params']['max_depth'] = int(a['params']['max_depth'])
a['params']['n_estimators'] = int(a['params']['n_estimators'])
print('最佳参数：', a)