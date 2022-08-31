
import numpy as np
from numpy import array
import seaborn as sns
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
import math
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import pythonscript



# Utilisez load_env pour tracer le chemin de .env:
""" destination = os.environ.get("destinationimportcollab")


with open(destination + '\\' + sys.argv[1], "r",encoding="ISO-8859-1") as file_obj:
 data = pd.read_csv(file_obj, skipinitialspace = True,on_bad_lines='skip',sep=';') """

#file_data["test"]=1
df=pythonscript.DataFrame

#print('22222222222222222')

#Regroupement par semaine

#La somme des pics vendus durant la semaine qui commence le Lundi indiqué (shipping_date)
#print('timeeeeeeeeeeeeeeeeeeeeeee')

df['shipping_date'] = pd.to_datetime(df['shipping_date'], infer_datetime_format=True) - pd.to_timedelta(6, unit='d')
df=df.resample('M', on='shipping_date').pics_number.sum()
df= pd.DataFrame(data=df)

df.drop(df[df['pics_number'] == 0].index, inplace = True)
dfw=df 


#MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
df_w_scaled=scaler.fit_transform(np.array(dfw).reshape(-1,1))

print('leen df_w_scaled : ')
print(len(df_w_scaled))
 
 
#splitting dataset into train and test split
training_size=int(len(df_w_scaled)*0.8)
test_size=len(df_w_scaled)-training_size
train_data,test_data=df_w_scaled[0:training_size,:],df_w_scaled[training_size:len(dfw),:1]
 
print('leen train : ')
print(len(train_data))

print('leen test : ')
print(len(test_data))
 
# convert an array of values into a dataset matrix
def create_dataset(dataset, time_step=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):
		a = dataset[i:(i+time_step), 0]   
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return np.array(dataX), np.array(dataY)
 

# reshape into X=t,t+1,t+2,t+3 and Y=t+4
time_step = 1
X_train, y_train = create_dataset(train_data, time_step)
X_test, ytest = create_dataset(test_data, time_step)

print('X_train : ')
print(X_train.shape)

print('X_test : ')
print(X_test.shape)
 
# reshape input to be [samples, time steps, features] which is required for LSTM
X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

""" print('****************')
print('X_train : ')
print(X_train.shape) """

#definir fonction MAE 
def mae(y_true, predictions):
  y_true, predictions = np.array(y_true), np.array(predictions)
  return np.mean(np.abs(y_true - predictions))
 
#definir fonction MAPE 
def MAPE(Y_actual,Y_Predicted):
  mape = np.mean(np.abs((Y_actual - Y_Predicted)/Y_actual))*100
  return mape


#declaration des variables



test = pd.DataFrame(columns=['parametres', 'mse_test', 'rmse_test', 'mae_test', 'mape_test'])
n_nodes = [1, 2]
epoch = [1]
batch_size = [1]
couche = [1, 2, 3, 4]



for c in couche:
  for i in n_nodes:
    for j in epoch:
      for k in batch_size:
       ''' print('n_couche:',c)
       print('n_nodes:',i)
       print('epoch:',j)
       print('batch:',k) '''
       model_test=Sequential()
       if c==1:
        model_test.add(LSTM(i,return_sequences=False,input_shape=(X_train.shape[1],X_train.shape[2])))
       else:
        model_test.add(LSTM(i,return_sequences=True,input_shape=(X_train.shape[1],X_train.shape[2])))

        t=0
        for t in range(c-2) :
         model_test.add(LSTM(i,return_sequences=True))
         t=t+1
        model_test.add(LSTM(i,return_sequences=False)) 
       model_test.add(Dense(1))
       model_test.compile(loss='mean_squared_error',optimizer='adam')
       history=model_test.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=j,batch_size=k,verbose=0)
       train_predict=model_test.predict(X_train)
       test_predict=model_test.predict(X_test)
      ##Transform back to original form
       train_predict=scaler.inverse_transform(train_predict)
       test_predict=scaler.inverse_transform(test_predict)

      #rmse_train= math.sqrt(mean_squared_error(y_train,train_predict))
      #print('rmse_train:',rmse_train)

       mse_test=mean_squared_error(ytest,test_predict)
       rmse_test=math.sqrt(mse_test)
       mae_test=mae(ytest, test_predict)
       mape_test= MAPE(ytest,test_predict)
       ''' print('mse_test : ',mse_test)
       print('rmse_test:',rmse_test)
       print('mae_test:',mae_test)
       print('mape_test:',mape_test) '''

       row = {'parametres': str(c)+'/'+str(i)+'/'+str(j)+'/'+str(k), 'mse_test':mse_test,'rmse_test':rmse_test,'mae_test':mae_test,'mape_test':mape_test }
       test = test.append(row, ignore_index = True)

#print(test)

 
""" # les combinaisons possibles
for c in couche:
    for i in n_nodes:
        for j in epoch:
            for k in batch_size:

                model_test = Sequential()
                if c == 1:
                    model_test.add(LSTM(i, return_sequences=False, input_shape=(
                        X_train.shape[1], X_train.shape[2])))
                else:
                    model_test.add(LSTM(i, return_sequences=True, input_shape=(
                        X_train.shape[1], X_train.shape[2])))
                    t = 0
                    for t in range(c-2):
                        model_test.add(LSTM(i, return_sequences=True))
                        t = t+1
                    model_test.add(LSTM(i, return_sequences=False))
                model_test.add(Dense(1))
                model_test.compile(loss='mean_squared_error', optimizer='adam')
                print('cccccc')
                model_test.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=j,batch_size=k,verbose=0)
                train_predict = model_test.predict(X_train)
                test_predict = model_test.predict(X_test)

                # Transform back to original form
                train_predict = scaler.inverse_transform(train_predict)
                test_predict = scaler.inverse_transform(test_predict)

                # error metrics
                mse_test = mean_squared_error(ytest, test_predict)
                rmse_test = math.sqrt(mse_test)
                mae_test = mae(ytest, test_predict)
                mape_test = MAPE(ytest, test_predict)

                row = {'parametres': str(c)+'/'+str(i)+'/'+str(j)+'/'+str(k), 'mse_test': mse_test,
                       'rmse_test': rmse_test, 'mae_test': mae_test, 'mape_test': mape_test}
                test = test.append(row, ignore_index=True)
print('test')
 """
# choisir les meilleurs parametres
test_trie = test.sort_values(
    by=['mse_test', 'rmse_test', 'mae_test', 'mape_test'])
best_param = test_trie['parametres'].iloc[0].split('/')


Couche = int(best_param[0])
neurone = int(best_param[1])
epochs = int(best_param[2])
batch = int(best_param[3]) 



#choisir les meilleurs parametres 
test_trie=test.sort_values(by=['mse_test','rmse_test','mae_test','mape_test'])
best_param=test_trie['parametres'].iloc[0].split('/')
  
""" print('best param :')
print(best_param) 
 """
Couche=int(best_param[0])
neurone=int(best_param[1])
epochs=int(best_param[2])
batch=int(best_param[3]) 


#Best LSTM Model
model_test = Sequential()
if Couche==1:
  model_test.add(LSTM(neurone,return_sequences=False,input_shape=(X_train.shape[1],X_train.shape[2])))
else:
  model_test.add(LSTM(i,return_sequences=True,input_shape=(X_train.shape[1],X_train.shape[2])))
  t=0
  for t in range(c-2):
    model_test.add(LSTM(i,return_sequences=True))
    t=t+1
  model_test.add(LSTM(i,return_sequences=False))
model_test.add(Dense(1))
model_test.compile(optimizer='adam', loss='mse')
history=model_test.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=epochs,batch_size=batch,verbose=0)

#Prediction 
train_predict=model_test.predict(X_train)
test_predict=model_test.predict(X_test)

##Transform back to original form
train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)
s=len(test_data)-1
x_input=test_data[s:].reshape(1,-1) #48

temp_input=list(x_input)
temp_input=temp_input[0].tolist() 

  
## predict next 4 weeks


lst_output=[]
n_steps=1
i=0
while(i<4):
    
    if(len(temp_input)>1):
        #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        #print(x_input)
        yhat = model_test.predict(x_input, verbose=False)
        print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model_test.predict(x_input, verbose=False)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1

mounth_pred=pd.DataFrame(scaler.inverse_transform(lst_output)) 
#print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
print('mounth_pred :',mounth_pred)
