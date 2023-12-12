import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc  ###计算roc和auc
import seaborn as sn

import xlrd
excel_path = './indicators/B01/DensenNet121.xlsX'
worksheet = xlrd.open_workbook(excel_path) # 打开xls文件
sheet_names= worksheet.sheet_names()

sheet = worksheet.sheet_by_name('val')
nrows = sheet.nrows # 行数
ncols = sheet.ncols # 列数
yscore_list = []
ytrue_list = []
for i in range(nrows):
    if i == 0:  # 跳过第一行
        continue
    ytrue = sheet.cell_value(i, 2)  # 取第二列数据
    yscore = sheet.cell_value(i, 5) # 取第四列数据

    yscore_list.append(yscore)
    ytrue_list.append(ytrue)

y_true = np.array(ytrue_list)
y_score = np.array(yscore_list)


# y_true:true label;
# y_score
fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score, pos_label=1)
roc_auc = auc(fpr,tpr) # 计算auc的值

threshold = 0.5 # 定义阈值
y_pred = [int(item>threshold) for  item in y_score]
score = metrics.accuracy_score(y_true, y_pred, normalize=True)

# 查准率
precision = metrics.precision_score(y_true, y_pred, average='macro')
# 查全率
recall = metrics.recall_score(y_true, y_pred, average='micro')
# F1 score
f1 = metrics.f1_score(y_true, y_pred, average='weighted')
print('Precision: {:.2f}%'.format(precision * 100))
print('Recall: {:.2f}%'.format(recall * 100))
print('F1: {:.2f}%'.format(f1 * 100))

# 混淆矩阵
confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 6),dpi=100)
sn.heatmap(confusion_matrix, annot=True)

# 分类报告：precision/recall/f1/分类个数
target_names = ['0(Normal)','1(CLL)']
classify_report = metrics.classification_report(y_true, y_pred,target_names=target_names)

overall_accuracy = metrics.accuracy_score(y_true, y_pred)
# 每类分类精度
acc_for_each_class = metrics.precision_score(y_true, y_pred, average=None)
average_accuracy = np.mean(acc_for_each_class)

print('classify_report : ', classify_report)
print('confusion_matrix : ', confusion_matrix)
print('acc_for_each_class : ', acc_for_each_class)
print('average_accuracy: {0:f}'.format(average_accuracy))
print('overall_accuracy: {0:f}'.format(overall_accuracy))
print('score: {0:f}'.format(score))

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
# plt.savefig("./results_img/train_vgg16_train.png")
plt.show()
