from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn import neighbors
from sklearn import svm

#[height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40], [190, 90, 47], [175, 64, 39],[177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]
Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female', 'female', 'male', 'male']

#DecisionTreeClassifier
clf = tree.DecisionTreeClassifier()

#GaussianNB
clf_gnb = GaussianNB()

#KNeighborsClassifier
clf_kneighbors = neighbors.KNeighborsClassifier()

#SVC
clf_svc = svm.SVC()


clf = clf.fit(X, Y)
clf_gnb=clf_gnb.fit(X,Y)
clf_kneighbors=clf_kneighbors.fit(X,Y)
clf_svc=clf_svc.fit(X,Y)

X_test=[[161,56,35]]
Y_test=['female']

prediction = clf.predict(X_test)
prediction1 = clf_gnb.predict(X_test)
prediction2 = clf_kneighbors.predict(X_test)
prediction3 = clf_svc.predict(X_test)

accuracy = clf.score(X_test, Y_test)
accuracy1 = clf_gnb.score(X_test, Y_test)
accuracy2 = clf_kneighbors.score(X_test, Y_test)
accuracy3 = clf_svc.score(X_test, Y_test)

print (prediction ,accuracy)
print (prediction1,accuracy1)
print (prediction2,accuracy2)
print (prediction3,accuracy3)

