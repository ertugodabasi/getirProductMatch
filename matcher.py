import pandas as pd
import re
import numpy as np
import featureExtractors as fe
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import fbeta_score
from sklearn.metrics import f1_score

df= pd.read_excel('getir.xlsx', sheet_name='getir')

df['levenshteinDistance'] = df.apply(fe.sorted_levenshtein_apply, axis=1)
df['uniqueNumberCount'] = df.apply(fe.get_unique_number_count, axis=1)+1
df['numberMatchRate'] = df.apply(fe.get_rate, axis=1)
df['matchScore'] = df.apply(fe.sorted_levenshtein_rate_apply, axis=1)
df['normalizedMatchRate'] = (df['numberMatchRate']+2).apply(np.log)
df['squaredPriceRate'] = df['priceRate']* df['priceRate']



X = df[['matchScore', 'squaredPriceRate', 'uniqueNumberCount', 'normalizedMatchRate']].values
y = df['Match'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,\
                                                    random_state=4, stratify=y)

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

#model = Perceptron(max_iter=40, eta0=0.1, random_state=1)
#model = LogisticRegression(C=100.0, random_state=1)
#model = SVC(kernel='linear', C=1.0, random_state=1)
#model = SVC(kernel='rbf', random_state=1, gamma=5.0, C=1.0)

models = {'Perceptron' : Perceptron(max_iter=40, eta0=0.1, random_state=1),
 'LogisticRegression' : LogisticRegression(C=100.0, random_state=1),
 'LinearSVC' : SVC(kernel='linear', C=1.0, random_state=1),
 'KernelizedSVC' : SVC(kernel='rbf', random_state=1, gamma=5.0, C=1.0),
 'MLP' : MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=(20, 20, 20), learning_rate='constant',
       learning_rate_init=0.001, max_iter=500, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=None,
       shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1,
       verbose=False, warm_start=False)}

for model_name, model in models.items():
    model.fit(X_train_std, y_train)
    y_pred = model.predict(X_test_std)
    print(model_name)
    print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
    print('Fscore: %.2f' % f1_score(y_test, y_pred))
#    print(classification_report(y_test, y_pred, labels=[1, 0], target_names=['match', 'no-match']))
    print('')


