import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics

# 设置绘图风格
# 获取所有的自带样式
# print (plt.style.available)
plt.style.use('ggplot')
# 设置中文编码和负号的正常显示
plt.rcParams['font.sans-serif'] = 'Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False

def save_image(x1,y1,x2,y2,X_axis,Y_axis,titleName,labels,imageName):

    # 设置图框的大小
    fig = plt.figure(figsize=(10, 6))
    # 绘图
    l1, = plt.plot(x1, y1, 'red',linewidth=2)
    l2, = plt.plot(x2, y2, 'blue',linewidth=2)

    # 添加轴标签和标题
    plt.title(titleName)
    plt.xlim(0, 1) # 限定横轴的范围
    plt.ylim(0, 1) # 限定纵轴的范围
    plt.xlabel(X_axis)
    plt.ylabel(Y_axis)

    # 显示图形
    # fig.autofmt_xdate(rotation=45)
    plt.legend(handles=[l1, l2, ], labels =labels)
    plt.savefig("./models_evalution/CLL_and_unCLL/" + imageName + ".jpg")
    # plt.show()

    return

# 导入数据
data_path = r'./models_evalution/CLL_and_unCLL/DensenNet121_add_25.xlsx'

df_train = pd.read_excel(data_path,sheet_name='train')
df_val = pd.read_excel(data_path,sheet_name='val')

# 画roc曲线
fpr_train, tpr_train, _ = metrics.roc_curve(df_train.y_true, df_train.y_positive_score, pos_label=1)
fpr_val, tpr_val, _ = metrics.roc_curve(df_val.y_true, df_val.y_positive_score, pos_label=1)
# 计算auc
auc_train = round(metrics.auc(fpr_train, tpr_train),4)
auc_val = round(metrics.auc(fpr_val, tpr_val),4)

# 画PR曲线
precision_train, recall_train, _ = metrics.precision_recall_curve(df_train.y_true, df_train.y_positive_score)
precision_val, recall_val, _ = metrics.precision_recall_curve(df_val.y_true, df_val.y_positive_score)


save_image(fpr_train,tpr_train,fpr_val,tpr_val,'False Positive Rate','True Positive Rate',\
           'train and val ROC',["train auc: "+ str(auc_train), "val auc: " + str(auc_val)],'DensenNet121_ROC_04')

save_image(recall_train,precision_train,recall_val,precision_val,'Recall','Precision',\
           'train and val PR',["train pr", "val pr"],'DensenNet121_PR_04')