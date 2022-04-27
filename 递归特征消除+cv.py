# ### 生成数据
# from sklearn.datasets import make_classification
# X, y = make_classification(n_samples=1000,         # 样本个数
#                            n_features=25,          # 特征个数
#                            n_informative=3,        # 有效特征个数
#                            n_redundant=2,          # 冗余特征个数（有效特征的随机组合）
#                            n_repeated=0,           # 重复特征个数（有效特征和冗余特征的随机组合）
#                            n_classes=8,            # 样本类别
#                            n_clusters_per_class=1, # 簇的个数
#                            random_state=0)
#
# ### 特征选择
# # RFE
# from sklearn.svm import SVC
# svc = SVC(kernel="linear")
#
# from sklearn.feature_selection import RFE
# rfe = RFE(estimator = svc,           # 基分类器
#           n_features_to_select = 2,  # 选择特征个数
#           step = 1,                  # 每次迭代移除的特征个数
#           verbose = 0                # 显示中间过程
#           ).fit(X,y)
# X_RFE = rfe.transform(X)
# print("RFE特征选择结果——————————————————————————————————————————————————")
# print("有效特征个数 : %d" % rfe.n_features_)
# print("全部特征等级 : %s" % list(rfe.ranking_))
#
# # RFECV
# from sklearn.svm import SVC
# svc = SVC(kernel="linear")
#
# from sklearn.model_selection import StratifiedKFold
# from sklearn.feature_selection import RFECV
# rfecv = RFECV(estimator=svc,          # 学习器
#               min_features_to_select=2, # 最小选择的特征数量
#               step=1,                 # 移除特征个数
#               cv=StratifiedKFold(2),  # 交叉验证次数
#               scoring='accuracy',     # 学习器的评价标准
#               verbose = 0,
#               n_jobs = 1
#               ).fit(X, y)
# X_RFECV = rfecv.transform(X)
# print("RFECV特征选择结果——————————————————————————————————————————————————")
# print("有效特征个数 : %d" % rfecv.n_features_)
# print("全部特征等级 : %s" % list(rfecv.ranking_))


from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn import metrics,tree
from xgboost import plot_importance,plot_tree
from yellowbrick.classifier import classification_report,ROCAUC,confusion_matrix
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np



# X,y = make_classification(n_samples=3000,         # 样本个数
#                            n_features=52,          # 特征个数
#                            n_informative=15,        # 有效特征个数
#                            n_redundant=2,          # 冗余特征个数（有效特征的随机组合）
#                            n_repeated=0,           # 重复特征个数（有效特征和冗余特征的随机组合）
#                            n_classes=7,            # 样本类别
#                            n_clusters_per_class=1, # 簇的个数
#                            random_state=0)

# X, y = iris.data, iris.target

data=pd.read_excel("重测数据总表 - 归一化前 - onlyfeature.xlsx")
X=data.drop(['TARGET','序号'],axis=1)
y=data['TARGET']


clf = XGBClassifier()
clf = clf.fit(X, y)
print(clf.feature_importances_)

model = SelectFromModel(clf,max_features=52, prefit=True,threshold=-np.inf)
X_new = model.transform(X)
print(X_new.shape[0])

print(X_new.shape)
ada=XGBClassifier()
x_train,x_test,y_train,y_test=train_test_split(X_new,y)
print(x_train)

ada.fit(x_train,y_train)
y_pre=ada.predict(x_test)

print(metrics.classification_report(y_test,y_pre))
# val=classification_report(ada,x_train,y_train,x_test,y_test)#可视化
# val1=confusion_matrix(ada,x_train,y_train,x_test,y_test)#可视化
# val2=ROCAUC(ada)#可视化

plot_importance(ada)
plt.show()


