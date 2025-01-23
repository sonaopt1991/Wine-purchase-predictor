import pandas as pd # type: ignore
import pickle


df=pd.read_csv("winequality-red.csv")

df['purchase']=(df['quality'] > 6).astype(int)

df['purchase'].value_counts(normalize=True)

x=df.drop(['quality','purchase'],axis=1)
y=df['purchase']

#remove imbalance in target column
from imblearn.over_sampling import SMOTE # type: ignore
sm=SMOTE(random_state=42)
x,y=sm.fit_resample(x,y)

#train test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

#rf
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(x_train,y_train)

pickle.dump(model,open('model.pkl','wb'))
