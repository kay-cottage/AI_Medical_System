from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
#from xgboost import XGBClassifier
import numpy as np
import warnings
warnings.filterwarnings("ignore")
# 这个方法只是解决了表面，没有根治

# 数据（特征，属性）
x_train = np.array([[1,2,3], 
          [1,5,4], 
          [2,2,2], 
          [4,5,6], 
          [3,5,4], 
          [1,7,2]]) 
# 数据的标签
y_train = np.array([1, 0, 1, 1, 0, 0]) 
 
# 测试数据
x_test = np.array([[2,1,2], 
          [3,2,6], 
          [2,6,4]]) 
 
# 导入模型
model = LogisticRegression() 
 
#model = RandomForestClassifier()

#model=XGBClassifier()

model.fit(x_train, y_train)

# 返回预测标签 
print(model.predict(x_test)) 
 
print('---------------------------------------')

# 返回预测属于某标签的概率 
print(model.predict_proba(x_test)) 
