import numpy as np
import pandas as pd

dangwei='26.5'
traindata = pd.read_excel('重测数据总表 - 归一化.xlsx',sheet_name=dangwei)
# print(traindata)
target = '分档'  # pose的值就是分类
x_columns = [x for x in traindata.columns if x not in [target]]
x_columns.append('分档')  # 得到标题列表


# 数据清洗
def harmonize_data(posedata):
    # 对数据进行归一化
    # 首先是归一化函数
    max_min_scaler = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
    # 我的数据集有38列，前36列为数值，第37列为时间，第38列为字符串类型，因此只对前36列做数值归一
    for title in x_columns[1:52]:
        posedata[title] = posedata[[title]].apply(max_min_scaler)
    # 把sit定义为0
    # posedata.loc[posedata['pose'] == 'sit', 'pose'] = 0
    return posedata


precessed_train_data = harmonize_data(traindata)
# print(precessed_train_data)
# precessed_train_data.to_excel('重测数据总表 - 0归一化.xlsx', index=False,sheet_name='0')


writer = pd.ExcelWriter('重测数据总表 - 0归一化.xlsx',mode='a', engine='openpyxl',if_sheet_exists='error')
precessed_train_data.to_excel(writer, sheet_name=dangwei)
writer.save()
writer.close()

