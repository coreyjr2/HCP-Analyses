import os
import pandas as pd
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
from IPython.display import Image

visualpath = "C:\\Users\\kyle\\repos\\HCP-Analyses\\Visuals\\"

cmap = {
  'Actual':"#604878", # Purple
  'Predicted':"#E84A27", # UIUC Orange
}

def write_scatter(label, dataframe, x, y, color, opacity, color_map = cmap, width=2000, height = 1200, trendline = 'lowess', trendline_options ={'frac':.1}, legend=True, yaxis_title = "Test Accuracy", showline=True, show_axes=True):
  fig  = px.scatter(dataframe, x=x,y=y,color=color, log_x=True,opacity=opacity,size_max =.01,color_discrete_map =color_map, width=width, height = height, trendline = trendline, trendline_options =trendline_options)
  fig.update_layout(
    showlegend=legend,
    xaxis_title="N Features",
    yaxis_title="Test Accuracy",
    legend_title="Data Source",
    font=dict(
        family="Times New Roman",
        size=50,
        color="Black"
        ),
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    yaxis_range=[0,1]
    )

  fig.update_yaxes(
    showgrid=False,
    #gridcolor="Black",
    showline=showline,
    linecolor="Black",
    visible=show_axes, showticklabels=show_axes
  )
  fig.update_xaxes(
    showline=showline,
    showgrid=False,
    linecolor="Black",
    visible=show_axes, showticklabels=show_axes
  )
  img_bytes = fig.to_image(format="png", width=600, height=350, scale=2)
  Image(img_bytes)
  # fig.show()
  fig.write_image(f'{visualpath}{label}.png')
  print(f'{visualpath}{label}.png')

base = f'S:\\'
sep = os.path.sep

hcp_path = f'{base}hcp_analysis_output{sep}89952a{sep}'
ucla_path = f'{base}ucla_analysis_output{sep}89952a{sep}'


hcp_output_path = f'{hcp_path}89952a_Prediction_Accuracies.csv'
# hcp_output_path2 = f'{hcp_path}Prediction_Accuracies.csv'
ucla_output_path = f'{ucla_path}Prediction_Accuracies.csv'

hcp_output = pd.read_csv(hcp_output_path)
# hcp_output2 = pd.read_csv(hcp_output_path2)
# hcp_output = pd.concat([hcp_output1,hcp_output2])
hcp_output.drop_duplicates(subset='metadata_ref', inplace=True)
ucla_output = pd.read_csv(ucla_output_path)
ucla_output.drop_duplicates(subset='metadata_ref', inplace=True)

output_df = pd.concat([hcp_output,ucla_output])
output_df.drop_duplicates(subset='metadata_ref', inplace=True)

output_df.to_csv(f'{base}HCP-UCLA_Prediction Accuracies.csv')

target_method = 'Random'


poly = PolynomialFeatures(degree=2)
x = hcp_output[hcp_output['FS/FR Method']==target_method][['N_Features']]
x_poly = poly.fit_transform(x)

y = hcp_output[hcp_output['FS/FR Method']==target_method]['test_accuracy']

regr = LinearRegression(fit_intercept=True)
regr.fit(x_poly, y)
regr.coef_

actual_df = hcp_output[hcp_output['FS/FR Method']==target_method][['N_Features','test_accuracy']]
actual_df['Source'] = 'Actual'

predictions = regr.predict(x_poly)
prediction_df = hcp_output[hcp_output['FS/FR Method']==target_method][['N_Features']]
prediction_df['Source'] = 'Predicted'
prediction_df['test_accuracy'] = predictions

plotting_df = pd.concat([actual_df, prediction_df])

print("Coefficients: \n", regr.coef_)
print("Mean squared error: %.2f" % mean_squared_error(y, predictions))
print("Coefficient of determination: %.2f" % r2_score(y, predictions))

op = .5
write_scatter(
    f'Ramdon_poly3_regression',
    plotting_df,
    x = 'N_Features',
    y='test_accuracy',
    color='Source',
    opacity=op,
    legend=False,
    trendline = None,
    trendline_options = None
  )

target_method = 'PCA'
target_dataset = 'UCLA'#'HCP_1200'
target_classifier = 'Support Vector Machine'
input_df = output_df[
  (output_df['dataset']==target_dataset) & 
  (output_df['FS/FR Method']==target_method) & 
  (output_df['Classifier']==target_classifier)
][['N_Features','test_accuracy']]
input_df['log(N_Features)'] = np.log(input_df['N_Features'])
input_df['sqrt(N_Features)'] = np.sqrt(input_df['N_Features'])
input_df['cbrt(N_Features)'] = np.cbrt(input_df['N_Features'])
# input_df['N_Features^2'] = input_df['N_Features']*input_df['N_Features']
# input_df['N_Features^3'] = input_df['N_Features']*input_df['N_Features']*input_df['N_Features']
input_df['Constant'] = 1
y = input_df['test_accuracy']

regr = LinearRegression(fit_intercept=True)
regr.fit(input_df[[
  'Constant'
  ,'N_Features'
  # ,'log(N_Features)'
  ,'sqrt(N_Features)'
  ,'cbrt(N_Features)'
  ]], y)
regr.coef_
predictions = regr.predict(input_df[[
  'Constant'
  ,'N_Features'
  # ,'log(N_Features)'
  ,'sqrt(N_Features)'
  ,'cbrt(N_Features)'
  ]])
print("Coefficients: \n", regr.coef_)
print("Mean squared error: %.2f" % mean_squared_error(y, predictions))
print("Coefficient of determination: %.2f" % r2_score(y, predictions))


actual_df = input_df[['N_Features','test_accuracy']]
actual_df['Source'] = 'Actual'

prediction_df = input_df[['N_Features']]
prediction_df['Source'] = 'Predicted'
prediction_df['test_accuracy'] = predictions

plotting_df = pd.concat([actual_df, prediction_df])

write_scatter(
  f'{target_dataset}_{target_classifier}_{target_method}FS_log+sqrt+cbrt',
  plotting_df,
  x = 'N_Features',
  y='test_accuracy',
  color='Source',
  opacity=op,
  legend=False,
  trendline = None,
  trendline_options = None
)

for target_method in output_df['FS/FR Method'].unique():
  for target_classifier in output_df['Classifier'].unique():
    for target_dataset in output_df['dataset'].unique():
      print(target_method, target_classifier, target_dataset)
      try:
        input_df = output_df[
          (output_df['dataset']==target_dataset) & 
          (output_df['FS/FR Method']==target_method) & 
          (output_df['Classifier']==target_classifier)
        ][['N_Features','test_accuracy']]
        input_df['log(N_Features)'] = np.log(input_df['N_Features'])
        input_df['sqrt(N_Features)'] = np.sqrt(input_df['N_Features'])
        input_df['cbrt(N_Features)'] = np.cbrt(input_df['N_Features'])
        input_df['Constant'] = 1
        y = input_df['test_accuracy']
        regr = LinearRegression(fit_intercept=True)
        regr.fit(input_df[[
          'Constant'
          ,'N_Features'
          ,'log(N_Features)'
          ,'sqrt(N_Features)'
          ,'cbrt(N_Features)'
          ]], y)
        regr.coef_
        predictions = regr.predict(input_df[[
          'Constant'
          ,'N_Features'
          ,'log(N_Features)'
          ,'sqrt(N_Features)'
          ,'cbrt(N_Features)'
          ]])
        
        print("\tCoefficients: \n\t", regr.coef_)
        print("\tMean squared error: %.2f" % mean_squared_error(y, predictions))
        print("\tCoefficient of determination: %.2f" % r2_score(y, predictions))
      except Exception as e:
        print(f'\t{e}')

# # def sigmoid(x, Beta_1, Beta_2):
# #   y = 1 / (1 + np.exp(-Beta_1*(x-Beta_2)))
# #   return y

# # def func(x, a, b, c):
# #   return a * np.exp(-b * x) + c

# def poly1(x, b1, b2, b3, c):
#   y = c + (b1*x) + (b2*(x^2)) + (b3*(x^3))
#   return y

# hcp_popt, hcp_pcov = curve_fit(
#   poly1,
#   hcp_output[hcp_output['FS/FR Method']==target_method]['N_Features'],
#   hcp_output[hcp_output['FS/FR Method']==target_method]['test_accuracy']
#   )

# plt.plot(
#   hcp_output[hcp_output['FS/FR Method']==target_method]['N_Features'],
#   poly1(
#     hcp_output[hcp_output['FS/FR Method']==target_method]['N_Features'],
#     *hcp_popt
#   ),
#   'r-',
#   label='fit: b1=%5.3f, b2=%5.3f, b3=%5.3f, c=%5.3f' % tuple(hcp_popt))

# plt.xlabel('x')

# plt.ylabel('y')

# plt.legend()

# plt.show()


cmap2 = {
  'Random':'#000000', # Black
  'Hierarchical Clustering':"#4EBE5D", # HCP green
  'PCA':"#E84A27", # UIUC Orange
  'Select From Model':"#EC2325",# HCP red
  'Permutation Importance':"#4877BC", # HCP blue
  'kPCA':"#4E8542", # Light Green
  'TruncatedSVD':"#F17CA9" # HCP
  # 'LDA':"#E84A27", # UIUC Orange
  # 'ICA':"#C19859" # Gold
}
op = .5
write_scatter(
  'SCV results HCP',
  output_df[
    (output_df['Classifier']=='Support Vector Machine') &
    (output_df['FS/FR Method']!='None') &
    (output_df['FS/FR Method']!='ICA') &
    (output_df['FS/FR Method']!='LDA') &
    (output_df['dataset']!='HCP_1200')
    ],
  x = 'N_Features',
  y='test_accuracy',
  color='FS/FR Method',
  opacity=op,
  color_map = cmap2
)