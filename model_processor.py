import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from matplotlib import cm
import json
class ModelProcessor:

  def __init__(self, df: pd.DataFrame, analysis_name: str, train_pct: float = 0.80, index_col = 'data', target_col = 'valoare', is_global: bool = False):
    if not 0 < train_pct < 1:
      raise ValueError("train_pct must be between 0 and 1 (e.g., 0.8 for 80%)")
    self.initial_df = df.copy()
    self.index_col = index_col
    self.target_col = target_col
    self.analysis_name = analysis_name
    self.is_global = is_global
    self.train_pct = train_pct
    self.scallers = {}
    self.scalled_df = None
    self.best_period = None
    self.prepare()
    if index_col not in self.initial_df.columns:
      raise ValueError(f"Index column: {index_col} does not exist in DataFrame")
    if target_col not in self.initial_df.columns:
      raise ValueError(f"Target column: {target_col} does not exist in DataFrame")
    self.train_df = None
    self.test_df = None
    self.split()
    self.results = {
      'index_col': self.index_col, 
      'target_col': self.target_col, 
      'analysis_name': self.analysis_name, 
      'is_global': self.is_global,
      'y_test': self.test_df[self.target_col].values,
      'y_test_unscalled': self.initial_df[self.initial_df[self.index_col].isin(self.test_df[self.index_col].values)][self.target_col].values,
      'predictions' : {}
    }
    plt.rcParams['figure.figsize'] = (14, 6)
    plt.style.use('seaborn-v0_8-whitegrid')


  def prepare(self) -> None:
    self.initial_df.columns = self.initial_df.columns.map(lambda x: x.lower().strip())
    self.initial_df[self.index_col] = pd.to_datetime(self.initial_df[self.index_col], format='ISO8601')
    self.initial_df.sort_values(self.index_col, inplace=True)
    scaller_class = StandardScaler if self.is_global else RobustScaler
    self.scalled_df = self.initial_df.copy()
    self.scallers = {col: scaller_class() for col in self.scalled_df.columns if col != self.index_col}
    for col in self.scalled_df.columns:
      if col != self.index_col:
        self.scalled_df[col] = self.scallers[col].fit_transform(self.scalled_df[[col]])

  def detect_acf_pacf(self, lags: int = 40) -> None:

    series = self.scalled_df[self.target_col].fillna(0)
    acf_vals = acf(series, nlags=lags)
    pacf_vals = pacf(series, nlags=lags)

    #todo : grafice separate
    #plt.figure(figsize=(14, 4))
    #ax1 = plt.subplot(121)
    #plot_acf(series, lags=lags, ax=ax1)
    #ax1.set_title(f'Autocorrelation Function - {self.analysis_name}', fontsize=16)
    
    #ax2 = plt.subplot(122)
    #plot_pacf(series, lags=lags, ax=ax2)
    #ax2.set_title(f'Partial Autocorrelation Function - {self.analysis_name}', fontsize=16)
    
    #plt.tight_layout()

    # === ACF ===
    plt.figure(figsize=(10, 4))
    plot_acf(series, lags=lags)
    plt.title(f'Autocorrelation Function - {self.analysis_name}', fontsize=16)
    plt.tight_layout()
    plt.show()

    # === PACF ===
    plt.figure(figsize=(10, 4))
    plot_pacf(series, lags=lags)
    plt.title(f'Partial Autocorrelation Function - {self.analysis_name}', fontsize=16)
    plt.tight_layout()
    plt.show()    
    plt.show()

    self.results['acf'] = acf_vals.tolist()
    self.results['pacf'] = pacf_vals.tolist()


  def detect_seasonality(self, periods: list = [12]) -> None:
    best_period = None
    best_resid_std = float('inf')
    resid_stds = {}
    decompositions = {}

    for period in periods:
      decomposition = seasonal_decompose(self.scalled_df[self.target_col].fillna(0), model='additive', period=period)
      resid_std = decomposition.resid.std()
      resid_stds[period] = resid_std
      decompositions[period] = decomposition
      if resid_std < best_resid_std:
        best_resid_std = resid_std
      best_period = period

    decomposition = decompositions[best_period]
    plt.figure(figsize=(16, 8))
    fig = decomposition.plot()
    #todo
    #fig.suptitle(f'Time Series Decomposition - {self.analysis_name} (Period={best_period})', fontsize=16)
    legend_labels = ['Residual'] + [f"p={p}: {resid_stds[p]:.4f}" for p in periods]    
    fig.legend(legend_labels, loc='upper left')
    plt.tight_layout()
    plt.show()

    self.results['seasonality'] = {
      'best_period': best_period,
      'residual_std': best_resid_std,
      'residual_stds': resid_stds
    }


  def display_analysis(self) -> None:

    if 'seasonality' in self.results:
      print(f"Optimal seasonality period: {self.results['seasonality']['best_period']}")
      print(f"Residual std dev: {self.results['seasonality']['residual_std']:.4f}")
    else:
      print("No seasonality detected.")

    if 'acf' in self.results and 'pacf' in self.results:
      max_acf_lag = max(range(1, len(self.results['acf'])), key=lambda i: abs(self.results['acf'][i]))
      max_pacf_lag = max(range(1, len(self.results['pacf'])), key=lambda i: abs(self.results['pacf'][i]))
      print(f"The largest ACF coefficient (excluding lag 0) is at lag {max_acf_lag}: {self.results['acf'][max_acf_lag]:.3f}")
      print(f"The largest PACF coefficient (excluding lag 0) is at lag {max_pacf_lag}: {self.results['pacf'][max_pacf_lag]:.3f}")
      if abs(self.results['acf'][1]) > 0.5:
          print("The series shows strong autocorrelation at the first lag (possible AR or MA process).")
      if abs(self.results['pacf'][1]) > 0.5:
          print("The series shows strong partial autocorrelation at the first lag (possible AR process).")
    else:
      print("ACF and PACF values were not calculated.")

  def analyze(self, periods: list = [12]) -> None:
    self.detect_seasonality(periods)
    self.detect_acf_pacf(lags=len(self.scalled_df) // 2)
    self.display_analysis()

  def split(self) -> None:
    train_size = int(len(self.initial_df) * self.train_pct)
    self.train_df = self.scalled_df.iloc[:train_size].reset_index(drop=True)
    self.test_df = self.scalled_df.iloc[train_size:].reset_index(drop=True)


  def run_model(self, model_name, model_func, **kwargs) -> None:

    results = model_func(self.train_df, self.test_df, self.index_col, self.target_col, **kwargs)
    if not isinstance(results, dict):
      raise ValueError("The model function must return a dictionary with the results")

    y_test = self.test_df[self.target_col].values
    y_pred = results.get('y_pred', None)
    if y_pred is None:
      raise ValueError("The model results must contain 'y_pred'")

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    r2 = r2_score(y_test, y_pred)   
    
    self.results['predictions'][model_name] = {
      'rmse': rmse,
      'mape': mape,
      'r2': r2,
      'mae': mae,
      'history': results.get('history', None),
      'y_pred': y_pred.reshape(-1, 1).flatten(),
      'y_unscaled': self.scallers[self.target_col].inverse_transform(y_pred.reshape(-1, 1)).flatten(),
    }

  @staticmethod
  def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict('records')
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif isinstance(obj, list):
        return [ModelProcessor.convert_to_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: ModelProcessor.convert_to_serializable(value) for key, value in obj.items()}
    elif hasattr(obj, 'item'):
        return obj.item()
    return obj
    
  def run_all_models(self, models: dict) -> None:
    for model_name, model_func in models.items():
      if not callable(model_func):
        raise ValueError(f"The function for model '{model_name}' is not valid")
      try:
        self.run_model(model_name, model_func)
      except Exception as e:
        print(f"Error running model {model_name}: {e}")
        self.results['predictions'][model_name] = {
          'error': str(e)
        }
    class NumpyEncoder(json.JSONEncoder):
      def default(self, obj):
          return ModelProcessor.convert_to_serializable(obj)
            
    with open(f'results_{self.analysis_name.replace(" ", "_").lower()}.json', 'w', encoding='utf-8') as f:
      json.dump(self.results, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

  def get_results(self) -> dict: 
    return self.results

  def process_results(self):
    res = self.results.get('predictions', {})
    if not res:
      return
    rows = []
    for k, v in res.items():
      t = 'multivariate' if 'multi' in k.lower() else 'univariate'
      rows.append({'model': k, 'type': t, 'r2': v['r2'], 'rmse': v['rmse'], 'mape': v['mape'], 'history': v['history']})
    df = pd.DataFrame(rows)

    uni = df[df['type'] == 'univariate'].sort_values('r2', ascending=False).reset_index(drop=True)
    print('\nUNIVARIATE')
    print(uni[['model', 'r2', 'rmse', 'mape']])

    set2_colors = [
      "#66c2a5",  # teal
      "#fc8d62",  # orange
      "#8da0cb",  # blue-purple
      "#e78ac3",  # pink
      "#448F44",  # forest green (în loc de verde aprins)
      "#DAA520",  # goldenrod (în loc de galben aprins)
      "#383a94"   # grey
    ]

    # === UNIVARIATE ===
    colors = set2_colors[:len(uni['model'])]
    if not uni.empty:
      for m in ['r2', 'rmse', 'mape']:
          plt.figure(figsize=(8, 5))
          plt.bar(uni['model'], uni[m], color=colors)
          plt.title(f'Univariate model performance - {self.analysis_name}\n{m.upper()}', fontsize=14)
          plt.xticks(rotation=45, ha='right')
          plt.tight_layout()
          plt.show()


    # === MULTIVARIATE ===
    multi = df[df['type'] == 'multivariate'].sort_values('r2', ascending=False).reset_index(drop=True)
    print('\nMULTIVARIATE')
    print(multi[['model', 'r2', 'rmse', 'mape']])
    colors = set2_colors[:len(multi['model'])]
    if not multi.empty:
      for m in ['r2', 'rmse', 'mape']:
          plt.figure(figsize=(8, 5))
          plt.bar(multi['model'], multi[m], color=colors)
          plt.title(f'Multivariate model performance - {self.analysis_name}\n{m.upper()}', fontsize=14)
          plt.xticks(rotation=45, ha='right')
          plt.tight_layout()
          plt.show()    
    # colors = cm.get_cmap('tab10', len(uni['model']))
    # if not uni.empty:
    #   fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    #   fig.suptitle(f'Univariate model performance - {self.analysis_name}', fontsize=16)
    #   for i, m in enumerate(['r2', 'rmse', 'mape']):
    #       bar_colors = [colors(j) for j in range(len(uni['model']))]
    #       ax[i].bar(uni['model'], uni[m], color=bar_colors)
    #       ax[i].set_title(m.upper())
    #       ax[i].tick_params(axis='x', rotation=45)
    #       for lab in ax[i].get_xticklabels():
    #         lab.set_horizontalalignment('right')
    #   plt.tight_layout(rect=[0, 0, 1, 0.95])


    # multi = df[df['type'] == 'multivariate'].sort_values('r2', ascending=False).reset_index(drop=True)
    # print('\nMULTIVARIATE')
    # print(multi[['model', 'r2', 'rmse', 'mape']])
    # colors = cm.get_cmap('tab10', len(multi['model']))
    # if not multi.empty:
    #   fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    #   fig.suptitle(f'Multivariate model performance - {self.analysis_name}', fontsize=16)
    #   for i, m in enumerate(['r2', 'rmse', 'mape']):
    #       bar_colors = [colors(j) for j in range(len(multi['model']))]
    #       ax[i].bar(multi['model'], multi[m], color=bar_colors)
    #       ax[i].set_title(m.upper())
    #       ax[i].tick_params(axis='x', rotation=45)
    #       for lab in ax[i].get_xticklabels():
    #         lab.set_horizontalalignment('right')
    #   plt.tight_layout(rect=[0, 0, 1, 0.95])


    if not uni.empty and not multi.empty:
        best_uni = uni.iloc[0]['model']
        best_multi = multi.iloc[0]['model']

        full_idx = pd.concat([self.train_df[self.index_col],
                              self.test_df[self.index_col]])
        full_real = pd.concat([self.train_df[self.target_col],
                               self.test_df[self.target_col]])

        pred_uni = pd.Series(np.nan, index=full_idx)
        pred_uni.loc[self.test_df[self.index_col]] = res[best_uni]['y_pred']

        pred_multi = pd.Series(np.nan, index=full_idx)
        pred_multi.loc[self.test_df[self.index_col]] = res[best_multi]['y_pred']

        plt.figure(figsize=(14, 6))
        plt.title(f'Predictions {self.analysis_name} - {best_uni} vs {best_multi}', fontsize=16)
        plt.plot(full_idx, full_real, label='real')
        plt.plot(full_idx, pred_uni, label=best_uni)
        plt.plot(full_idx, pred_multi, label=best_multi)

        plt.axvspan(self.test_df[self.index_col].min(),
                    self.test_df[self.index_col].max(),
                    alpha=0.1, color='grey')

        plt.legend()
        plt.tight_layout()
        plt.show()

    #todo : ordonare dupa denumirea modelului
    #hist = df[df['history'].notnull()].sort_values('model', ascending=False)
    hist = df[df['history'].notnull()].copy()
    hist['model_name'] = hist['model'].str.replace(r'^(Univariate|Multivariate)\s+', '', regex=True)
    hist = hist.sort_values('model_name', ascending=False)
    if not hist.empty:
        n, cols = len(hist), 2
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 7, rows * 4), squeeze=False)
        axes = axes.flatten()

        for idx, row in hist.reset_index().iterrows():
            h = res[row['model']]['history']
            if h and 'loss' in h:
                axes[idx].plot(h['loss'])
                axes[idx].set_title(row['model'])
                axes[idx].set_xlabel('epoch')
                axes[idx].set_ylabel('loss')

        for ax in axes[n:]:
            fig.delaxes(ax)
        plt.suptitle(f'Loss evolution over epochs - {self.analysis_name}', fontsize=16)
        plt.tight_layout()
        plt.show()
