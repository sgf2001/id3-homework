#导入相关库
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import export_graphviz
import graphviz
import treeplotter
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


"读取红酒数据集"
inputfile='C:/go/winequality-red.csv'
f = open(inputfile,"r")

xList = []
labels = []
feature_name = []
firstLine = True
for line in f:
    if firstLine:
        feature_name = line.strip().split(",")
        firstLine = False
    else:
        row = line.strip().split(",")
        labels.append(float(row[-1]))
        row.pop()
        floatRow = [float(num) for num in row]
        xList.append(floatRow)

#nrows = len(xList)
#ncols = len(xList[0])
#print(nrows)
#print(ncols)

#print(xList)
#print(labels)
#print(feature_name)

X_train, X_test, Y_train, Y_test = train_test_split(
    xList, labels, test_size=0.30,random_state=0)


# 定义决策树类
model = tree.DecisionTreeClassifier(max_depth=8,random_state=30,criterion="entropy",splitter="random")

# 数据、训练
model.fit(X_train,Y_train)

#预测
print('训练得分：', model.score(X_train, Y_train))
print('测试得分：', model.score(X_test, Y_test))


feature_name = ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']
labels_name = ['0','0','0','1','1','1']
export_graphviz(
    model,
    out_file="./tree.dot",
    feature_names=feature_name,
    class_names=labels_name,
    rounded=True,
    filled=True
)


with open("./tree.dot",encoding='UTF-8') as f:
    dot_grapth = f.read()
dot = graphviz.Source(dot_grapth)
dot.view()
print([*zip(feature_name,model.feature_importances_)]) #查看特征向量的重要性

'''绘制决策边界'''

axes=[5.0,10.0,0.2,1.0]
matrix = np.array(xList)
X = matrix[:,:2]#这是两个指标
y = labels

model = tree.DecisionTreeClassifier(max_depth=8,random_state=30,min_samples_leaf=5
                ,min_samples_split=5,criterion="entropy",splitter="random")

# 数据、训练
model.fit(X,y)
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure()
plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.show()


'''确定决策树深度'''

train= []
test = []
for i in range(100):
    clf = tree.DecisionTreeClassifier(max_depth=i+1,criterion="entropy",random_state=30,splitter="random")
    clf = clf.fit(X_train,Y_train)
    score = clf.score(X_test, Y_test)
    score_1 = clf.score(X_train,Y_train)
    test.append(score)
    train.append(score_1)

plt.plot(range(1,101),test,color="red",label="max_depth")
plt.plot(range(1,101),train,color="blue",label="max_depth")
plt.legend()
plt.show()#8

















