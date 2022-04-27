import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# from sklearn.metrics import classification_report

data = pd.read_excel('yes-no.xlsx')

# X=data[['Volume', 'Pra', 'Prb', 'Pouter_radius', 'Pinner_radius', 'Parea', 'Pcontlength', 'Pdist_mean', 'Prect2_len2', 'Pmoments_m20', 'Pmoments_ib', 'Pmoments_i1', 'Pmoments_i3', 'Pmoments_i4']]##null-importance
X=data[['Volume', 'Pra', 'Prb', 'Pouter_radius', 'Pinner_radius', 'Parea', 'Pcontlength', 'Pdist_mean', 'Prect2_len2', 'Pmoments_m20', 'Pmoments_ib',  'Pphi','Porientation','Area','Length2','Pmax_diameter','max_temp_z',
        'Length3','Prect2_len1','Length1','Pinner_width','Pinner_height','Pmoments_ia','Pmoments_m02']]##方差和Kendall系数的特征选择前24
# X=data.drop(['Pmoments_psi2','Pmoments_m12_invar','Pmoments_psi4','Pmoments_m03_invar','Pmoments_m30_invar',
#              'Pmoments_m03','Pmoments_m21','Pmoments_m12','Pmoments_m30',
#              '序号', '质量', '人工测量：长', '人工测量：高', '分档', '宽1', '宽2', '宽3', '人工测量：宽', '边1', '边2', '边3', '备注'],axis=1)#方差和Kendall系数的特征选择前45
# X=data.drop(['TARGET','序号'],axis=1)
print((len(X.columns)))
print(X.head())
Y=data['分档']



x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.3,random_state=42)

from xgboost.sklearn import XGBClassifier
model=XGBClassifier(n_estimators=200,learning_rate=0.09)
model.fit(x_train, y_train)
y_pred=model.predict(x_test)
print(accuracy_score(y_test,y_pred))
