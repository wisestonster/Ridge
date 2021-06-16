# https://www.codeit.kr/learn/courses/machine-learning/3353
# 정규화(Regularization)
# 12. L2 정규화 직접 해보기

from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from math import sqrt

import numpy as np
import pandas as pd

INSURANCE_FILE_PATH = './datasets/insurance.csv'

insurance_df = pd.read_csv(INSURANCE_FILE_PATH)
insurance_df = pd.get_dummies(data=insurance_df, columns=['sex', 'smoker', 'region'])

# 기존 데이터에서 입력 변수 데이터를 따로 저장한다
X = insurance_df.drop(['charges'], axis=1)

polynomial_transformer = PolynomialFeatures(4)  # 4 차항 변형기를 정의한다
polynomial_features = polynomial_transformer.fit_transform(X.values)  # 데이터 6차 항 변수로 바꿔준다

features = polynomial_transformer.get_feature_names(X.columns)  # 변수 이름들도 바꿔준다

# 새롭게 만든 다항 입력 변수를 dataframe으로 만들어 준다
X = pd.DataFrame(polynomial_features, columns=features)
y = insurance_df[['charges']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5)

model = Ridge(alpha=0.01, max_iter=2000, normalize=True)
model.fit(X_train, y_train)

y_train_predict = model.predict(X_train)
y_test_predict = model.predict(X_test)

# 체점용 코드
mse = mean_squared_error(y_train, y_train_predict)

print("training set에서 성능")
print("-----------------------")
print(f'오차: {sqrt(mse)}')

mse = mean_squared_error(y_test, y_test_predict)

print("testing set에서 성능")
print("-----------------------")
print(f'오차: {sqrt(mse)}')