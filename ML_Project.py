
from matplotlib import pyplot as plt
from itertools import cycle
import pandas as pd 
import numpy as np 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize


#states = ['CA']
drop_cols = ['End_Lat', 'End_Lng', 'Number', 'Precipitation(in)', 'Wind_Chill(F)']
df = pd.read_csv('US_Accidents_June20.csv', nrows=5000)
#df = df[df.State.isin(states)]
df = df.drop(drop_cols, axis=1)
df = df.dropna()
df.info()
print(df.head(40))

df_dummy = pd.get_dummies(df)
df_dummy.info()
df = df_dummy

def plot_roc(y, y_t, pred):
	n_classes = y.shape[1]
	fpr = dict()
	tpr = dict()
	roc_auc = dict()
	for i in range(n_classes):
	    fpr[i], tpr[i], _ = roc_curve(y_t[:, i], pred[:, i])
	    roc_auc[i] = auc(fpr[i], tpr[i])
	colors = cycle(['red', 'green', 'Blue', 'Yellow'])
	lw=1
	for i, color in zip(range(n_classes), colors):
	    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
	             label='ROC curve of class {0} (area = {1:0.2f})'
	             ''.format(i+1, roc_auc[i]))

	plt.plot([0, 1], [0, 1], 'k--', lw=lw)
	plt.xlim([0, 1])
	plt.ylim([0, 1])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic for multi-class data')
	plt.legend(loc="lower right")
	plt.show()

def print_repot_cm(y_t, pred):
	y_test_cm = [ np.argmax(t) for t in y_t ]
	y_predict_cm = [ np.argmax(t) for t in pred ]
	print('\nAccuracy: {:.2f}'.format(accuracy_score(y_t, pred.round())))
	print('Precision: {:.2f}'.format(precision_score(y_t, pred.round(), average='micro')))
	print('Recall: {:.2f}'.format(recall_score(y_t, pred.round(), average='micro')))
	print('ROC-AUC: {:.2f}\n'.format(roc_auc_score(y_t, pred.round(), average='micro')))
	cm = confusion_matrix(y_test_cm, y_predict_cm)
	sns.heatmap(cm, annot=True, annot_kws = {"size": 16})
	plt.show()

y = df['Severity']
X = df.drop('Severity', axis=1)
y = label_binarize(y, classes=[1, 2, 3, 4])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

dt = OneVsRestClassifier(DecisionTreeClassifier())
dt.fit(X_train, y_train)
dt_pred = dt.predict_proba(X_test)
plot_roc(y, y_test, dt_pred)
print_repot_cm(y_test, dt_pred)

rf = OneVsRestClassifier(RandomForestRegressor(n_estimators=10))
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
plot_roc(y, y_test, rf_pred)
print_repot_cm(y_test, rf_pred)

knn = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=3))
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)
plot_roc(y, y_test, knn_pred)
print_repot_cm(y_test, knn_pred)

lr = OneVsRestClassifier(LogisticRegression())
lr.fit(X_train, y_train)
lr_pred = lr.predict_proba(X_test)
plot_roc(y, y_test, lr_pred)
print_repot_cm(y_test, lr_pred)