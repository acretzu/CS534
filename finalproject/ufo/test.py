import csv
import datetime
import random
import time
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

X = []  # data
y = []  # classes (shape)

with open('ufo-sightings/scrubbed.csv', newline='') as csvfile:
    data = list(csv.reader(csvfile))

# Correct bad data
for row in data[1:]:

    if len(row) == 11:

        if row[0][9:] == "24:00":
            str = row[0][:9] + "00:00"
        elif row[0][10:] == "24:00":
            str = row[0][:10] + "00:00"
        elif row[0][11:] == "24:00":
            str = row[0][:11] + "00:00"
        else:
            str = row[0]

        dt = datetime.datetime.strptime(str, '%m/%d/%Y %H:%M')
        sec = time.mktime(dt.timetuple())
        yrs = sec/60/60/24/7/52

        try:
            X.append( [#yrs,               # yrs since 1970
                      float(row[5])])    # duration (s)
                      #float(row[9]),    # latitude
                      #float(row[10])])  # longitude

            y.append(row[4])            # shape
        except:
            print("ignoring data in row:", row)

random.shuffle(X)

# Turn shape strings into numbered classes
classes = ['cylinder', 'light', 'circle',
           'sphere', 'disk', 'fireball',
           'unknown', 'oval', 'other',
           'cigar', 'rectangle', 'chevron',
           'triangle', 'formation', '',
           'delta', 'changing', 'egg',
           'flash', 'diamond', 'cross',
           'teardrop', 'cone', 'pyramid',
           'round', 'crescent', 'flare',
           'hexagon', 'dome', 'changed']

# for i in range(len(y)):
#     for j in range(len(classes)):
#         if y[i] == classes[j]:
#             y[i] = j

print("Named classes: ", classes)
print("Data classes: ", y[:100])
print("Data: ", X[:100])

# Train and Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=2)
knn = KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train)
score = knn.score(X_test, y_test)
print(score)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=2)
# lr = Lasso().fit(X_train, y_train)
# score = lr.score(X_train, y_train)
# print(score)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=2)
# nn = MLPClassifier().fit(X_train, y_train)
# score = nn.score(X_test, y_test)
# print(score)