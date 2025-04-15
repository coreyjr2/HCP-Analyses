import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import Image

# Read coefficients
function_df = pd.read_csv('C:\\Users\\kyle\\repos\\HCP-Analyses\\Dimensionality_Reduction_Models.csv')
function_df.fillna(0, inplace=True)
prediction_df = pd.DataFrame({"N_Features":range(1,19900)})
prediction_df['input_size'] = 1670
prediction_df['Dataset'] = 'UCLA (1670)'

prediction_dfs = []
for method in function_df['Method'].unique():
  method_df = prediction_df.copy()
  method_df['Method'] = method
  intercept = list(function_df.loc[function_df['Method']==method,'X.Intercept.'])[0]
  input_size_loading = list(function_df.loc[function_df['Method']==method,'input_size'])[0]
  n_loading = list(function_df.loc[function_df['Method']==method,'N_Features'])[0]
  n2_loading = list(function_df.loc[function_df['Method']==method,'N_Features_sqrt'])[0]
  n3_loading = list(function_df.loc[function_df['Method']==method,'N_Features_cbrt'])[0]
  method_df['PredictedAccuracy'] = (
    intercept + 
    (input_size_loading*prediction_df['input_size']) + 
    (prediction_df['N_Features']*n_loading) + 
    ((prediction_df['N_Features']**(1./2))*n2_loading) + 
    ((prediction_df['N_Features']**(1./3))*n3_loading)
  )
  prediction_dfs.append(method_df)

prediction_df_2 = pd.concat(prediction_dfs)

# fig1 = px.line(prediction_df_2, x='N_Features', y='PredictedAccuracy', color='Method', log_x=True, range_y=(0, 1), color_discrete_sequence=px.colors.qualitative.G10)
# fig1.show()

prediction_df['input_size'] = 6677
prediction_df['Dataset'] = 'HCP-1200 (6677)'

for method in function_df['Method'].unique():
  method_df = prediction_df.copy()
  method_df['Method'] = method
  intercept = list(function_df.loc[function_df['Method']==method,'X.Intercept.'])[0]
  input_size_loading = list(function_df.loc[function_df['Method']==method,'input_size'])[0]
  n_loading = list(function_df.loc[function_df['Method']==method,'N_Features'])[0]
  n2_loading = list(function_df.loc[function_df['Method']==method,'N_Features_sqrt'])[0]
  n3_loading = list(function_df.loc[function_df['Method']==method,'N_Features_cbrt'])[0]
  method_df['PredictedAccuracy'] = (
    intercept + 
    (input_size_loading*prediction_df['input_size']) + 
    (prediction_df['N_Features']*n_loading) + 
    ((prediction_df['N_Features']**(1./2))*n2_loading) + 
    ((prediction_df['N_Features']**(1./3))*n3_loading)
  )
  prediction_dfs.append(method_df)

prediction_df_3 = pd.concat(prediction_dfs)

# fig2 = px.line(prediction_df_3, x='N_Features', y='PredictedAccuracy', color='Method', log_x=True, range_y=(0, 1), line_dash='input_size', color_discrete_sequence=px.colors.qualitative.G10)
# fig2.show()

prediction_df_3.rename(columns={'N_Features':'N Features'}, inplace=True)

prediction_df_3.loc[prediction_df_3['Method'].str.contains('final_model_HC.coefficients'),'Method'] = 'Hierarchical Clustering'
prediction_df_3.loc[prediction_df_3['Method'].str.contains('final_model_kPCA.coefficients'),'Method'] = 'kPCA'
prediction_df_3.loc[prediction_df_3['Method'].str.contains('final_model_PCA.coefficients'),'Method'] = 'PCA'
prediction_df_3.loc[prediction_df_3['Method'].str.contains('final_model_SFM.coefficients'),'Method'] = 'Select from Model'
prediction_df_3.loc[prediction_df_3['Method'].str.contains('final_model_tSVD.coefficients'),'Method'] = 'tSVD'
prediction_df_3.loc[prediction_df_3['Method'].str.contains('final_model_PI.coefficients'),'Method'] = 'FRPI'

fig3 = px.line(prediction_df_3, x='N Features', y='PredictedAccuracy', color='Method', log_x=True, range_y=(0, 1), line_dash='Dataset', color_discrete_sequence=px.colors.qualitative.G10)
fig3.update_layout(
  height=1000, 
  width=800, 
  # title_text=f'Network Connection Feature Importance',
  showlegend=True,
  legend=dict(
    orientation='v',
    yanchor="bottom",
    y=-.5,
    xanchor="center",
    x=.2
  ),
  font=dict(
    family="Arial",
    size=12,
    color="Black"
    ),
  template='simple_white',
  paper_bgcolor='White',
  plot_bgcolor='White'
)
fig3.show()
fig3.write_image('AccuracyCurves.png')