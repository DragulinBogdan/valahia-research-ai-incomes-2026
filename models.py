import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.vecm import VECM
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
from keras.api.models import Sequential
from keras.api.layers import Dense, LSTM, GRU, Conv1D, Flatten
from keras.api.optimizers import Adam

def ETSUnivariate(train_df, test_df, index_col, target_col, seasonal_periods=12):
    train_series = train_df[target_col].values
    model = ExponentialSmoothing(
        train_series,
        seasonal='add',  # additive seasonality
        seasonal_periods=seasonal_periods,
        trend='add',  # additive trend
        initialization_method='estimated'
    )
    fitted = model.fit(optimized=True, use_brute=True)
    y_pred = fitted.forecast(steps=len(test_df))
    
    return {'y_pred': y_pred}

def ProphetUnivariate(train_df, test_df, index_col, target_col):
    train_prophet = train_df[[index_col, target_col]].copy()
    train_prophet.columns = ['ds', 'y']
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,  # no weekly for monthly data
        daily_seasonality=False,
        seasonality_mode='additive'
    )
    model.fit(train_prophet)
    future = pd.DataFrame({
        'ds': test_df[index_col]
    })
    forecast = model.predict(future)
    y_pred = forecast['yhat'].values
    
    return {'y_pred': y_pred}

def SeasonalNaiveUnivariate(train_df, test_df, index_col, target_col, season_length=12):
    train_series = train_df[target_col].values    
    last_season = train_series[-season_length:]    
    n_periods = len(test_df)
    y_pred = np.tile(last_season, n_periods // season_length + 1)[:n_periods]
    return {'y_pred': y_pred}

def LinearRegressionUnivariate(train_df, test_df, index_col, target_col):
    X_train=train_df[[index_col]].copy()
    y_train=train_df[[target_col]].copy()
    X_test=test_df[[index_col]].copy()
    for df in (X_train,X_test):
        df['month']=df[index_col].dt.month
        df['year']=df[index_col].dt.year
        df['trend']=np.arange(len(df))
        df.drop(columns=[index_col],inplace=True)
    m=LinearRegression().fit(X_train,y_train)
    y_pred=m.predict(X_test)
    return {'y_pred':y_pred}

def LinearRegressionMultivariate(train_df, test_df, index_col, target_col):
    cols=[c for c in train_df.columns if c not in (index_col,target_col)]
    if not cols:
        cols=['trend']
        train_df['trend']=np.arange(len(train_df))
        test_df['trend']=np.arange(len(test_df))
    X_train=train_df[cols].values
    y_train=train_df[target_col].values
    X_test=test_df[cols].values
    m=LinearRegression().fit(X_train,y_train)
    y_pred=m.predict(X_test)
    return {'y_pred':y_pred}

def SARIMAUnivariate(train_df, test_df, index_col, target_col, order=(1,1,1),seasonal_order=(1,1,1,12)):
    m=SARIMAX(train_df[target_col],order=order,seasonal_order=seasonal_order).fit(disp=False)
    y_pred=m.forecast(len(test_df))
    return {'y_pred':y_pred.values}

def SARIMAMultivariate(train_df, test_df, index_col, target_col, order=(1,1,1),seasonal_order=(1,1,1,12)):
    ex_cols=[c for c in train_df.columns if c not in (index_col,target_col)]
    ex_train=train_df[ex_cols] if ex_cols else None
    ex_test=test_df[ex_cols] if ex_cols else None
    m=SARIMAX(train_df[target_col],exog=ex_train,order=order,seasonal_order=seasonal_order).fit(disp=False)
    y_pred=m.forecast(len(test_df),exog=ex_test)
    return {'y_pred':y_pred.values}

def VARMultivariate(train_df, test_df, index_col, target_col,maxlags = 10):
    cols=[c for c in train_df.columns if c!=index_col]
    model=VAR(train_df[cols])
    res=model.fit(maxlags=maxlags,ic='aic')
    fc=res.forecast(train_df[cols].values[-res.k_ar:],len(test_df))
    y_pred=fc[:,cols.index(target_col)]
    return {'y_pred':y_pred}

def ECMMultivariate(train_df, test_df, index_col, target_col):
    cols=[c for c in train_df.columns if c!=index_col]
    vec=VECM(train_df[cols],k_ar_diff=1,coint_rank=1).fit()
    fc=vec.predict(len(test_df))
    y_pred=fc[target_col].values if isinstance(fc,pd.DataFrame) else fc[:,cols.index(target_col)]
    return {'y_pred':y_pred}

def _seq_uni(arr,look_back):
    X,y=[],[]
    for i in range(len(arr)-look_back):
        X.append(arr[i:i+look_back])
        y.append(arr[i+look_back])
    return np.array(X),np.array(y)

def _seq_multi(arr,targets,look_back):
    X,y=[],[]
    for i in range(len(arr)-look_back):
        X.append(arr[i:i+look_back])
        y.append(targets[i+look_back])
    return np.array(X),np.array(y)

def LSTMUnivariate(train_df, test_df, index_col, target_col,look_back=12,epochs=50):
    train_series=train_df[target_col].values
    X_train,y_train=_seq_uni(train_series,look_back)
    X_train=X_train.reshape((X_train.shape[0],X_train.shape[1],1))
    model=Sequential([LSTM(64,input_shape=(look_back,1)),Dense(1)])
    model.compile(optimizer=Adam(),loss='mse')
    history=model.fit(X_train,y_train,epochs=epochs,verbose=0,batch_size=32)
    full=np.concatenate([train_series[-look_back:],test_df[target_col].values])
    X=[full[i:i+look_back] for i in range(len(test_df))]
    X=np.array(X).reshape((len(test_df),look_back,1))
    y_pred=model.predict(X).flatten()
    return {'y_pred':y_pred,'history':history.history}

def GRUUnivariate(train_df, test_df, index_col, target_col,look_back=12,epochs=50):
    train_series=train_df[target_col].values
    X_train,y_train=_seq_uni(train_series,look_back)
    X_train=X_train.reshape((X_train.shape[0],X_train.shape[1],1))
    model=Sequential([GRU(64,input_shape=(look_back,1)),Dense(1)])
    model.compile(optimizer=Adam(),loss='mse')
    history=model.fit(X_train,y_train,epochs=epochs,verbose=0,batch_size=32)
    full=np.concatenate([train_series[-look_back:],test_df[target_col].values])
    X=[full[i:i+look_back] for i in range(len(test_df))]
    X=np.array(X).reshape((len(test_df),look_back,1))
    y_pred=model.predict(X).flatten()
    return {'y_pred':y_pred,'history':history.history}

def TCNUnivariate(train_df, test_df, index_col, target_col,look_back=12,epochs=50):
    train_series=train_df[target_col].values
    X_train,y_train=_seq_uni(train_series,look_back)
    X_train=X_train.reshape((X_train.shape[0],look_back,1))
    model=Sequential([Conv1D(64,2,activation='relu',dilation_rate=1,input_shape=(look_back,1)),Conv1D(64,2,activation='relu',dilation_rate=2),Flatten(),Dense(1)])
    model.compile(optimizer=Adam(),loss='mse')
    history=model.fit(X_train,y_train,epochs=epochs,verbose=0,batch_size=32)
    full=np.concatenate([train_series[-look_back:],test_df[target_col].values])
    X=[full[i:i+look_back] for i in range(len(test_df))]
    X=np.array(X).reshape((len(test_df),look_back,1))
    y_pred=model.predict(X).flatten()
    return {'y_pred':y_pred,'history':history.history}

def LSTMMultivariate(train_df, test_df, index_col, target_col,look_back=12,epochs=50):
    feat_cols=[c for c in train_df.columns if c not in (index_col,target_col)]
    if not feat_cols:
        feat_cols=[target_col]
    train_X=train_df[feat_cols].values
    train_y=train_df[target_col].values
    X_train,y_train=_seq_multi(train_X,train_y,look_back)
    X_train=X_train.reshape((X_train.shape[0],look_back,len(feat_cols)))
    model=Sequential([LSTM(64,input_shape=(look_back,len(feat_cols))),Dense(1)])
    model.compile(optimizer=Adam(),loss='mse')
    history=model.fit(X_train,y_train,epochs=epochs,verbose=0,batch_size=32)
    full_X=np.vstack([train_X[-look_back:],test_df[feat_cols].values])
    X=[full_X[i:i+look_back] for i in range(len(test_df))]
    X=np.array(X).reshape((len(test_df),look_back,len(feat_cols)))
    y_pred=model.predict(X).flatten()
    return {'y_pred':y_pred,'history':history.history}

def GRUMultivariate(train_df, test_df, index_col, target_col,look_back=12,epochs=50):
    feat_cols=[c for c in train_df.columns if c not in (index_col,target_col)]
    if not feat_cols:
        feat_cols=[target_col]
    train_X=train_df[feat_cols].values
    train_y=train_df[target_col].values
    X_train,y_train=_seq_multi(train_X,train_y,look_back)
    X_train=X_train.reshape((X_train.shape[0],look_back,len(feat_cols)))
    model=Sequential([GRU(64,input_shape=(look_back,len(feat_cols))),Dense(1)])
    model.compile(optimizer=Adam(),loss='mse')
    history=model.fit(X_train,y_train,epochs=epochs,verbose=0,batch_size=32)
    full_X=np.vstack([train_X[-look_back:],test_df[feat_cols].values])
    X=[full_X[i:i+look_back] for i in range(len(test_df))]
    X=np.array(X).reshape((len(test_df),look_back,len(feat_cols)))
    y_pred=model.predict(X).flatten()
    return {'y_pred':y_pred,'history':history.history}

def TCNMultivariate(train_df, test_df, index_col, target_col, look_back=12, epochs=50):
    feat_cols = [c for c in train_df.columns if c not in (index_col, target_col)] or [target_col]
    train_X, train_y = train_df[feat_cols].values, train_df[target_col].values

    # secvențe (look-back)
    X_train, y_train = _seq_multi(train_X, train_y, look_back)
    X_train = X_train.reshape((X_train.shape[0], look_back, len(feat_cols)))

    model = Sequential([
        Conv1D(64, 2, activation='relu', dilation_rate=1,
               input_shape=(look_back, len(feat_cols))),
        Conv1D(64, 2, activation='relu', dilation_rate=2),
        Flatten(),
        Dense(1)
    ])
    model.compile(optimizer=Adam(), loss='mse')
    history = model.fit(X_train, y_train, epochs=epochs,
                        verbose=0, batch_size=32)

    # prognoză
    full_X = np.vstack([train_X[-look_back:], test_df[feat_cols].values])
    X = [full_X[i:i + look_back] for i in range(len(test_df))]
    X = np.array(X).reshape((len(test_df), look_back, len(feat_cols)))
    y_pred = model.predict(X).flatten()

    return {'y_pred': y_pred, 'history': history.history}

def ToateModelele():
    return {
        'Univariate Linear Regression': LinearRegressionUnivariate,
        'Multivariate Linear Regression': LinearRegressionMultivariate,
        'Univariate SARIMA': SARIMAUnivariate,
        'Multivariate SARIMA': SARIMAMultivariate,
        'Univariate LSTM': LSTMUnivariate,
        'Multivariate LSTM': LSTMMultivariate,
        'Univariate GRU': GRUUnivariate,
        'Multivariate GRU': GRUMultivariate,
        'Univariate TCN': TCNUnivariate,
        'Multivariate TCN': TCNMultivariate,
        'Multivariate VAR': VARMultivariate,
        'Multivariate ECM': ECMMultivariate,
        'Univariate ETS': ETSUnivariate,
        'Univariate Prophet': ProphetUnivariate,
        'Univariate Seasonal Naive': SeasonalNaiveUnivariate
    }