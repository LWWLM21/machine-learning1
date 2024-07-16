import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.utils import Bunch
import seaborn as sns
def MAPE(true, pred):
    diff = np.abs(np.array(true) - np.array(pred))
    return np.mean(diff / true)
df = pd.read_excel('20240704DATASET1.xlsx')
data_column=['mc','$K_s$']
data_feature=['mc','$K_s$']
bunch = Bunch(
  data=df.drop(data_feature,axis=1).values,  # 特征数据
  target=df['$K_s$'].values,       # 目标变量
  feature_names=df.drop(data_feature, axis=1).columns.tolist(),  # 特征名称列表
  target_names=['$K_s$']  # 目标变量名称列表
)
X=pd.DataFrame(bunch.data,columns=bunch.feature_names)
y=bunch.target
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=1)
model=GradientBoostingRegressor(random_state=123,n_estimators=823,max_depth=2,learning_rate=0.39743)#max_leaf_nodes=5, min_samples_leaf=1, min_samples_split=2
model.fit(X_train,y_train)
Z7 = model.predict(X_train)
Z8 = model.predict(X_test)
print("GBRT Training R2:", r2_score(y_train, Z7), "RMSE:", np.sqrt(mean_squared_error(y_train, Z7)), "MAE:", mean_absolute_error(y_train, Z7),"MAPE:", MAPE(y_train, Z7))
print("GBRT Testing R2:", r2_score(y_test, Z8), "RMSE:", np.sqrt(mean_squared_error(y_test, Z8)), "MAE:", mean_absolute_error(y_test, Z8),"MAPE:", MAPE(y_test, Z8))
xx = np.linspace(-1, 120, 1000)
yy = xx
sns.set_style("whitegrid")
sns.set(style="ticks")
plt.rcParams['font.sans-serif']=['Times New Roman']
plt.figure(figsize=(8,6))

plt.rcParams['font.sans-serif']=['Times New Roman']
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.plot(xx, yy, '--',c='grey', linewidth=1.5)
plt.scatter(y_train, Z7,marker='o')
plt.scatter(y_test, Z8,c='darkorange',marker='s')

plt.tick_params (axis='both',labelsize=15)
plt.yticks(fontproperties = 'Times New Roman', size = 15)
plt.xticks(fontproperties = 'Times New Roman', size = 15)

font1 = {'family' : 'Times New Roman', 'weight' : 'normal', 'size' : 20,}
plt.axis('tight')
plt.xlabel('Tested Ks(kN/mm)',fontdict={'family': 'Times New Roman','size':16})
plt.ylabel('Predicted Ks(kN/mm)',fontdict={'family': 'Times New Roman','size':16})
plt.xlim([-1,120])
plt.ylim([-1,120])
plt.title('GradientBoostingRegressor',fontdict={'family': 'Times New Roman','size':16})
plt.legend(['y = x','Training set','Testing set'], loc = 'upper left', prop={'family' : 'Times New Roman', 'weight' : 'normal', 'size' : 14,})
plt.savefig('梯度提升.jpg')
plt.show()
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)


shap.summary_plot(shap_values, X, plot_type="dot", show=True)
plt.ylabel(r'$\mathit{italic\_text}_\mathrm{subscript}$',fontdict={'family': 'Times New Roman','size':16}) # 使用LaTeX语法设置斜体和下标

plt.show()
