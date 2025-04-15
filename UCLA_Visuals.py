import umap.umap_ as umap
import pandas as pd
import datetime as dt
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import Image
import os
import numpy as np

cmap2 = {
  'Random':"#FFFFFF", # White
  'Hierarchical Clustering':"#9F2936",
  # 'PCA':"#6A76FC",
  # 'Select From Model':"#1B587C",
  # 'Permutation Importance':"#FE00CE",
  'Eigenvector':"#4E8542", # Light Green
  'Participation':"#604878", # Purple
  'Betweenness':"#E84A27", # UIUC Orange
  'Degree':"#C19859", # Gold
  'Pagerank':"#EEA6FB"
}

cmap_diagnosis = {
  'CONTROL':'rgb(127, 60, 141)',#"#FFFFFF", 
  'SCHZ':'rgb(17, 165, 121)',#"#4E8542",
  'BIPOLAR':'rgb(57, 105, 172)',#"#EEA6FB",
  'ADHD':'rgb(242, 183, 1)'#"#E84A27"
}

cmap_task = {
  'scap':'#9F2936',
  'bart':'#6A76FC', 
  'taskswitch':'#1B587C', 
  'stopsignal':'#FE00CE', 
  'rest':'#4E8542', 
  'bht':'#604878',
  'pamret':'#EEA6FB', 
  'pamenc':'#C19859'
}

cmap_gender = {
  'M':"#6A76FC", 
  'F':"#EEA6FB",
}

session_list = [
  'scap','bart','pamret','pamenc',
  'taskswitch','stopsignal','rest','bht'
  ]

uid_dict = {
  '69354adf':{
    'n_parcels':200,
    'AROMA':'No AROMA'
  },
  '4023aba1':{
    'n_parcels':200,
    'AROMA':'smoothAROMAnonaggr'
  },
  '5384cc6b':{
    'n_parcels':1000,
    'AROMA':'No AROMA'
  },
  '8a52e148':{
    'n_parcels':1000,
    'AROMA':'smoothAROMAnonaggr'
  }
}

def connection_names(corr_matrix, labels):
  name_idx = np.triu_indices_from(corr_matrix, k=1)
  out_list = []
  for i in range(len(name_idx[0])):
    out_list.append(str(labels[name_idx[0][i]]) + '|' + str(labels[name_idx[1][i]]))
  return out_list

parcellation_path = 'S:\\Code\\MSC_HCP\\Parcellation\\'#'/data/hx-hx1/kbaacke/Code/Parcellation/'
#trying different stuff
n_list = [2, 5, 10, 20, 50, 100, 200]
d_list = [0.0, 0.1, 0.25, 0.5, 0.8, 0.99]
n_list_str = []
for n in n_list:
  n_list_str.append(str(n))

d_list_str = []
for d in d_list:
  d_list_str.append(str(d))

sep = os.path.sep
ucla_path = 'S:\\UCLA\\'

if False: # Example plot to test options
  uid = '69354adf'
  n_parcels = uid_dict[uid]['n_parcels']
  parcel_info = pd.read_csv(f'{parcellation_path}Schaefer2018_{n_parcels}Parcels_7Networks_order_FSLMNI152_2mm.Centroid_RAS.csv')
  node_names = list(parcel_info['ROI Name'])
  dummy_array = np.zeros((n_parcels,n_parcels))
  colnames = connection_names(dummy_array, node_names)
  session='rest'
  session_path = f'{ucla_path}{uid}{sep}{session}{sep}'
  session_plt = make_subplots(
    rows=len(n_list), 
    cols=len(d_list), 
    row_titles=n_list_str, 
    column_titles=d_list_str,
    x_title = 'Minimum Distance',
    y_title = 'N Neighbors'
  )
  try:
    os.mkdir(session_path)
  except:
    pass
  try:
    os.mkdir(session_path + f'diagnosis{sep}')
  except:
    pass
  try:
    os.mkdir(session_path + f'gender{sep}')
  except:
    pass
  try:
    os.mkdir(session_path + f'age{sep}')
  except:
    pass
  session_data = pd.DataFrame(np.load(f'{ucla_path}{uid}{sep}{uid}_{session}_FunctionalConnectomes.npy', allow_pickle=True), columns = np.load(f'{ucla_path}{uid}{sep}{uid}_{session}_FunctionalConnectomes_colnames.npy', allow_pickle=True))
  scaled_session_data = StandardScaler().fit_transform(session_data[colnames])
  n = 5
  d = 0.1
  fit = umap.UMAP(
    n_neighbors=n,
    min_dist=d,
    n_components=2,
    metric='euclidean'
  )
  umap_output = fit.fit_transform(scaled_session_data)
  umap_df = pd.DataFrame(umap_output)
  umap_df.columns = ['x','y']
  umap_df['diagnosis'] = session_data['diagnosis']
  umap_df['participant_id'] = session_data['participant_id']
  umap_df['age'] = session_data['age'].astype(int)
  umap_df['gender'] = session_data['gender']
  umap_df['ScannerSerialNumber'] = session_data['ScannerSerialNumber']
  umap_df.to_csv(f'{session_path}{uid}_{session}_{n}-neighbors_d-{d}.csv', index=False)
  scatter_diagnosis_1 = px.scatter(umap_df, x='x',y='y',color='diagnosis',opacity=1,size_max =.01,color_discrete_map =cmap_diagnosis, width=2000, height = 1200)
  scatter_diagnosis_1.update_layout(
    showlegend=True,
    xaxis_title="X",
    yaxis_title="Y",
    legend_title="Diagnosis",
    font=dict(
      family="Times New Roman",
      size=30,
      color="Grey"
      ),
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
  )
  scatter_diagnosis_1.update_yaxes(
    showline=True,
    showgrid=False,
    linecolor="Grey",
    visible=True, 
    showticklabels=True,
    zeroline=False
  )
  scatter_diagnosis_1.update_xaxes(
    showline=True,
    showgrid=False,
    linecolor="Grey",
    visible=True, 
    showticklabels=True,
    zeroline=False
  )
  img_bytes = scatter_diagnosis_1.to_image(format="png", width=600, height=600, scale=2)
  Image(img_bytes)
  scatter_diagnosis_1.write_image(f'{session_path}{uid}_{session}_{n}-neighbors_d-{d}_diagnosis.png')

  scatter_age_1 = px.scatter(
    umap_df, x='x',y='y',color='age',opacity=1,size_max =.01,width=2000, height = 1200, color_continuous_scale = 'Inferno'
  )
  scatter_age_1.update_layout(
    showlegend=True,
    xaxis_title="X",
    yaxis_title="Y",
    legend_title="Diagnosis",
    font=dict(
      family="Times New Roman",
      size=30,
      color="Grey"
      ),
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
  )
  scatter_age_1.update_yaxes(
    showline=True,
    showgrid=False,
    linecolor="Grey",
    visible=True, 
    showticklabels=True,
    zeroline=False
  )
  scatter_age_1.update_xaxes(
    showline=True,
    showgrid=False,
    linecolor="Grey",
    visible=True, 
    showticklabels=True,
    zeroline=False
  )
  img_bytes = scatter_age_1.to_image(format="png", width=600, height=600, scale=2)
  Image(img_bytes)
  scatter_age_1.write_image(f'{session_path}{uid}_{session}_{n}-neighbors_d-{d}_age.png')

  if False: # Example subplotting
    ex_01 =  make_subplots(
      rows=5, 
      cols=5, 
      row_titles=['0','1','2','3','4'], 
      column_titles=['0','1','2','3','4'],
      x_title = 'Minimum Distance',
      y_title = 'N Neighbors'
    )
    for i in range(5):
      for j in range(5):
        for dat in scatter_diagnosis_1.data:
          if i==0 and j==0:
            ex_01.add_trace(
              go.Scatter(
                x=dat['x'],
                y=dat['y'],
                name=umap_df['name'],
                opacity=1,
                # color_discrete_map =cmap_diagnosis,
                legendgroup=dat['name'],
                showlegend=True,
                marker={'color':cmap_diagnosis[dat['name']]},
                mode='markers'
              ),
              row=i+1,
              col=j+1,
            )
          else:
            ex_01.add_trace(
              go.Scatter(
                x=dat['x'],
                y=dat['y'],
                name=umap_df['name'],
                opacity=1,
                # color_discrete_map =cmap_diagnosis,
                legendgroup=dat['name'],
                showlegend=False,
                marker={'color':cmap_diagnosis[dat['name']]},
                mode='markers'
              ),
              row=i+1,
              col=j+1,
            )

    ex_01.update_layout(
      height=2000, 
      width=2000, 
      title_text=f'Diagnosis UMAP {uid}',
      showlegend=True,
      font=dict(
        family="Times New Roman",
        # size=30,
        color="Grey"
        ),
      template='plotly_dark',
      paper_bgcolor='rgba(0,0,0,0)',
      plot_bgcolor='rgba(0,0,0,0)'
    )

    ex_01.update_yaxes(
      showgrid=False,
      #gridcolor="Black",
      showline=False,
      linecolor="Grey",
      visible=True, 
      showticklabels=True,
      title={
        'font':dict(
          family="Times New Roman",
          # size=30,
          color="Grey"
        )
      },
      zeroline=False
    )

    ex_01.update_xaxes(
      showline=True,
      showgrid=False,
      linecolor="Grey",
      visible=True, 
      showticklabels=True,
      title={
        'font':dict(
          family="Times New Roman",
          # size=30,
          color="Grey"
        )
      },
      zeroline=False
        
    )
    ex_01.show()
    img_bytes = ex_01.to_image(format="png", width=2000, height=2000, scale=1)
    Image(img_bytes)
    ex_01.write_image(f'{session_path}ex1.png')
    ex_01.write_html(f'{session_path}ex1.html')


    ex_02 =  make_subplots(
      rows=5, 
      cols=5, 
      row_titles=['0','1','2','3','4'], 
      column_titles=['0','1','2','3','4'],
      x_title = 'Minimum Distance',
      y_title = 'N Neighbors'
    )
    for i in range(5):
      for j in range(5):
        for dat in scatter_age_1.data:
          if i==0 and j==0:
            ex_02.add_trace(
              go.Scatter(
                x=dat['x'],
                y=dat['y'],
                # name='Age',
                opacity=1,
                # color_discrete_map =cmap_diagnosis,
                # legendgroup=dat['name'],
                showlegend=False,
                marker={
                  'colorscale':'Inferno',
                  'color':dat['marker']['color'],
                  'colorbar':{
                    'title':'Age',
                  }
                },
                mode='markers'
              ),
              row=i+1,
              col=j+1,
            )
          else:
            ex_02.add_trace(
              go.Scatter(
                x=dat['x'],
                y=dat['y'],
                name='Age',
                opacity=1,
                # color_discrete_map =cmap_diagnosis,
                # legendgroup=dat['name'],
                showlegend=False,
                marker={
                  'colorscale':'Inferno',
                  'color':dat['marker']['color']
                  },
                mode='markers'
              ),
              row=i+1,
              col=j+1,
            )

    ex_02.update_layout(
      height=2000, 
      width=2000, 
      title_text=f'Age UMAP {uid}',
      showlegend=True,
      font=dict(
        family="Times New Roman",
        # size=30,
        color="Grey"
        ),
      template='plotly_dark',
      paper_bgcolor='rgba(0,0,0,0)',
      plot_bgcolor='rgba(0,0,0,0)'
    )

    ex_02.update_yaxes(
      showgrid=False,
      #gridcolor="Black",
      showline=True,
      linecolor="Grey",
      visible=True, 
      showticklabels=True,
      title={
        'font':dict(
          family="Times New Roman",
          # size=30,
          color="Grey"
        )
      },
      zeroline=False
    )

    ex_02.update_xaxes(
      showline=True,
      showgrid=False,
      linecolor="Grey",
      visible=True, 
      showticklabels=True,
      title={
        'font':dict(
          family="Times New Roman",
          # size=30,
          color="Grey"
        )
      },
      zeroline=False 
    )
    ex_02.show()
    img_bytes = ex_02.to_image(format="png", width=2000, height=2000, scale=1)
    Image(img_bytes)
    ex_02.write_image(f'{session_path}ex2.png')
    ex_02.write_html(f'{session_path}ex2.html')

full_fig_list = []
# Diagnosis, Age, and Gender UMAPs within each task
if False:
  for uid in list(uid_dict.keys())[2:]:
    n_parcels = uid_dict[uid]['n_parcels']
    parcel_info = pd.read_csv(f'{parcellation_path}Schaefer2018_{n_parcels}Parcels_7Networks_order_FSLMNI152_2mm.Centroid_RAS.csv')
    node_names = list(parcel_info['ROI Name'])
    dummy_array = np.zeros((n_parcels,n_parcels))
    colnames = connection_names(dummy_array, node_names)
    uid_fig_list = []
    for session in session_list:
      session_path = f'{ucla_path}{uid}{sep}{session}{sep}'
      session_plt_diagnosis = make_subplots(
        rows=len(n_list), 
        cols=len(d_list), 
        row_titles=n_list_str, 
        column_titles=d_list_str,
        x_title = 'Minimum Distance',
        y_title = 'N Neighbors'
      )
      session_plt_age = make_subplots(
        rows=len(n_list), 
        cols=len(d_list), 
        row_titles=n_list_str, 
        column_titles=d_list_str,
        x_title = 'Minimum Distance',
        y_title = 'N Neighbors'
      )
      session_plt_gender = make_subplots(
        rows=len(n_list), 
        cols=len(d_list), 
        row_titles=n_list_str, 
        column_titles=d_list_str,
        x_title = 'Minimum Distance',
        y_title = 'N Neighbors'
      )
      # Make output directories
      if True:
        try:
          os.mkdir(session_path)
        except:
          pass
        try:
          os.mkdir(session_path + f'diagnosis{sep}')
        except:
          pass
        try:
          os.mkdir(session_path + f'gender{sep}')
        except:
          pass
        try:
          os.mkdir(session_path + f'age{sep}')
        except:
          pass
      session_data = pd.DataFrame(np.load(f'{ucla_path}{uid}{sep}{uid}_{session}_FunctionalConnectomes.npy', allow_pickle=True), columns = np.load(f'{ucla_path}{uid}{sep}{uid}_{session}_FunctionalConnectomes_colnames.npy', allow_pickle=True))
      session_data = session_data[~session_data[colnames].isna().any(axis=1)]
      scaled_session_data = StandardScaler().fit_transform(session_data[colnames])
      for n in n_list:
        for d in d_list:
          try:
            umap_df = pd.read_csv(f'{session_path}{uid}_{session}_{n}-neighbors_d-{d}_UMAP.csv')
          except:
            fit = umap.UMAP(
              n_neighbors=n,
              min_dist=d,
              n_components=2,
              metric='euclidean'
            )
            umap_output = fit.fit_transform(scaled_session_data)
            umap_df = pd.DataFrame(umap_output)
            umap_df.columns = ['x','y']
            umap_df['diagnosis'] = session_data['diagnosis']
            umap_df['participant_id'] = session_data['participant_id']
            umap_df['age'] = session_data['age'].astype(int)
            umap_df['gender'] = session_data['gender']
            umap_df['ScannerSerialNumber'] = session_data['ScannerSerialNumber']
            umap_df = umap_df[~pd.isna(umap_df['diagnosis'])]
            umap_df.to_csv(f'{session_path}{uid}_{session}_{n}-neighbors_d-{d}_UMAP.csv', index=False)
          # Output Individual images
          if True:
            # Diagnosis
            if True:
              scatter_diagnosis_1 = px.scatter(
                umap_df, 
                x='x',
                y='y',
                color='diagnosis',
                opacity=1,
                size_max =.01,
                color_discrete_map =cmap_diagnosis,
                width=2000, 
                height = 1200
              )
              scatter_diagnosis_1.update_layout(
                showlegend=True,
                xaxis_title="X",
                yaxis_title="Y",
                legend_title="Diagnosis",
                font=dict(
                  family="Times New Roman",
                  size=30,
                  color="Grey"
                  ),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
              )
              scatter_diagnosis_1.update_yaxes(
                showgrid=False,
                #gridcolor="Black",
                showline=False,
                linecolor="Grey",
                visible=True, showticklabels=True,
                zeroline=False
              )
              scatter_diagnosis_1.update_xaxes(
                showline=True,
                showgrid=False,
                linecolor="Grey",
                visible=True, showticklabels=True,
                zeroline=False
              )
              img_bytes = scatter_diagnosis_1.to_image(format="png", width=600, height=600, scale=2)
              Image(img_bytes)
              scatter_diagnosis_1.write_image(f'{session_path}diagnosis{sep}{uid}_{session}_{n}-neighbors_d-{d}_diagnosis_UMAP.png')
            #Age
            if True:
              scatter_age_1 = px.scatter(
                umap_df, 
                x='x',
                y='y',
                color='age',
                opacity=1,
                size_max =.01, 
                width=2000, 
                height = 1200, 
                color_continuous_scale = 'Inferno'
              )
              scatter_age_1.update_layout(
                showlegend=True,
                xaxis_title="X",
                yaxis_title="Y",
                legend_title="Age",
                font=dict(
                  family="Times New Roman",
                  size=30,
                  color="Grey"
                  ),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
              )
              scatter_age_1.update_yaxes(
                showgrid=False,
                #gridcolor="Black",
                showline=False,
                linecolor="Grey",
                visible=True, showticklabels=True,
                zeroline=False
              )
              scatter_age_1.update_xaxes(
                showline=True,
                showgrid=False,
                linecolor="Grey",
                visible=True, showticklabels=True,
                zeroline=False
              )
              img_bytes = scatter_age_1.to_image(format="png", width=600, height=600, scale=2)
              Image(img_bytes)
              scatter_age_1.write_image(f'{session_path}age{sep}{uid}_{session}_{n}-neighbors_d-{d}_age_UMAP.png')
            # Gender
            if True:
              scatter_gender_1 = px.scatter(
                umap_df, 
                x='x',
                y='y',
                color='gender',
                opacity=1,
                size_max =.01,
                color_discrete_map =cmap_gender, 
                width=2000, 
                height = 1200
              )
              scatter_gender_1.update_layout(
                showlegend=True,
                xaxis_title="X",
                yaxis_title="Y",
                legend_title="Gender",
                font=dict(
                  family="Times New Roman",
                  size=30,
                  color="Grey"
                  ),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
              )
              scatter_gender_1.update_yaxes(
                showgrid=False,
                #gridcolor="Black",
                showline=False,
                linecolor="Grey",
                visible=True, showticklabels=True,
                zeroline=False
              )
              scatter_gender_1.update_xaxes(
                showline=True,
                showgrid=False,
                linecolor="Grey",
                visible=True, showticklabels=True,
                zeroline=False
              )
              img_bytes = scatter_gender_1.to_image(format="png", width=600, height=600, scale=2)
              Image(img_bytes)
              scatter_gender_1.write_image(f'{session_path}gender{sep}{uid}_{session}_{n}-neighbors_d-{d}_gender_UMAP.png')
          # Add subplots to array figures
          if True:
            if d_list.index(d)==0 and n_list.index(n)==0:
              show_legend = True
            else:
              show_legend = False
            # diagnosis
            for dat in scatter_diagnosis_1.data:
              session_plt_diagnosis.add_trace(
                go.Scatter(
                  x=dat['x'],
                  y=dat['y'],
                  name=dat['name'],
                  opacity=1,
                  # color_discrete_map =cmap_diagnosis,
                  legendgroup=dat['name'],
                  showlegend=show_legend,
                  marker={'color':cmap_diagnosis[dat['name']]},
                  mode='markers'
                ),
                row=n_list.index(n)+1,
                col=d_list.index(d)+1,
              )
            # age
            for dat in scatter_age_1.data:
              if show_legend:
                session_plt_age.add_trace(
                  go.Scatter(
                    x=dat['x'],
                    y=dat['y'],
                    opacity=1,
                    showlegend=False,
                    marker={
                      'colorscale':'Inferno',
                      'color':dat['marker']['color'],
                      'colorbar':{
                        'title':'Age',
                      }
                    },
                    mode='markers'
                  ),
                  row=n_list.index(n)+1,
                  col=d_list.index(d)+1,
                )
              else:
                session_plt_age.add_trace(
                go.Scatter(
                    x=dat['x'],
                    y=dat['y'],
                    opacity=1,
                    showlegend=False,
                    marker={
                      'colorscale':'Inferno',
                      'color':dat['marker']['color']
                    },
                    mode='markers'
                  ),
                  row=n_list.index(n)+1,
                  col=d_list.index(d)+1,
                )
            # gender
            for dat in scatter_gender_1.data:
              session_plt_gender.add_trace(
                go.Scatter(
                  x=dat['x'],
                  y=dat['y'],
                  name=dat['name'],
                  opacity=1,
                  # color_discrete_map =cmap_diagnosis,
                  legendgroup=dat['name'],
                  showlegend=show_legend,
                  marker={'color':cmap_gender[dat['name']]},
                  mode='markers'
                ),
                row=n_list.index(n)+1,
                col=d_list.index(d)+1,
              )
      aroma = uid_dict[uid]['AROMA']
      session_plt_diagnosis.update_layout(
        height=2000, 
        width=2000, 
        title_text=f'Diagnosis UMAP {session}, {n_parcels} parcels, {aroma}',
        showlegend=True,
        font=dict(
          family="Times New Roman",
          # size=30,
          color="Grey"
          ),
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
      )
      session_plt_diagnosis.update_yaxes(
        showgrid=False,
        #gridcolor="Black",
        showline=True,
        linecolor="Grey",
        visible=True, 
        showticklabels=True,
        title={
          'font':dict(
            family="Times New Roman",
            # size=30,
            color="Grey"
          )
        },
        zeroline=False
      )
      session_plt_diagnosis.update_xaxes(
        showline=True,
        showgrid=False,
        linecolor="Grey",
        visible=True, 
        showticklabels=True,
        title={
          'font':dict(
            family="Times New Roman",
            # size=30,
            color="Grey"
          )
        },
        zeroline=False
      )
      session_plt_age.update_layout(
        height=2000, 
        width=2000, 
        title_text=f'Age UMAP {session}, {n_parcels} parcels, {aroma}',
        showlegend=True,
        font=dict(
          family="Times New Roman",
          # size=30,
          color="Grey"
          ),
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
      )
      session_plt_age.update_yaxes(
        showgrid=False,
        #gridcolor="Black",
        showline=True,
        linecolor="Grey",
        visible=True, 
        showticklabels=True,
        title={
          'font':dict(
            family="Times New Roman",
            # size=30,
            color="Grey"
          )
        },
        zeroline=False
      )
      session_plt_age.update_xaxes(
        showline=True,
        showgrid=False,
        linecolor="Grey",
        visible=True, 
        showticklabels=True,
        title={
          'font':dict(
            family="Times New Roman",
            # size=30,
            color="Grey"
          )
        },
        zeroline=False
      )
      session_plt_gender.update_layout(
        height=2000, 
        width=2000, 
        title_text=f'Gender UMAP {session}, {n_parcels} parcels, {aroma}',
        showlegend=True,
        font=dict(
          family="Times New Roman",
          # size=30,
          color="Grey"
          ),
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
      )
      session_plt_gender.update_yaxes(
        showgrid=False,
        #gridcolor="Black",
        showline=True,
        linecolor="Grey",
        visible=True, 
        showticklabels=True,
        title={
          'font':dict(
            family="Times New Roman",
            # size=30,
            color="Grey"
          )
        },
        zeroline=False
      )
      session_plt_gender.update_xaxes(
        showline=True,
        showgrid=False,
        linecolor="Grey",
        visible=True, 
        showticklabels=True,
        title={
          'font':dict(
            family="Times New Roman",
            # size=30,
            color="Grey"
          )
        },
        zeroline=False
      )
      session_fig_list = [
        session_plt_diagnosis,
        session_plt_age,
        session_plt_gender
      ]
      session_plt_diagnosis.write_image(f'{session_path}diagnosis{sep}{uid}_{session}_diagnosis_UMAP.png')
      session_plt_age.write_image(f'{session_path}age{sep}{uid}_{session}_age_UMAP.png')
      session_plt_gender.write_image(f'{session_path}gender{sep}{uid}_{session}_gender_UMAP.png')
      session_plt_diagnosis.write_html(f'{session_path}diagnosis{sep}{uid}_{session}_diagnosis_UMAP.html')
      session_plt_age.write_html(f'{session_path}age{sep}{uid}_{session}_age_UMAP.html')
      session_plt_gender.write_html(f'{session_path}gender{sep}{uid}_{session}_gender_UMAP.html')
      with open(f'{session_path}{uid}_{session}_UMAP.html', 'a') as f:
        for fig in session_fig_list:
          uid_fig_list.append(fig)
          full_fig_list.append(fig)
          f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))
    with open(f'{ucla_path}{uid}{sep}{uid}_UMAP.html', 'a') as f:
      for fig in uid_fig_list:
        f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))
  with open(f'{ucla_path}UMAP-FULL.html', 'a') as f:
    for fig in full_fig_list:
      f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))

# Task decoding UMAP
task_fig_list = []
for uid in list(uid_dict.keys())[2:]:
  # uid = '69354adf'
  n_parcels = uid_dict[uid]['n_parcels']
  parcel_info = pd.read_csv(f'{parcellation_path}Schaefer2018_{n_parcels}Parcels_7Networks_order_FSLMNI152_2mm.Centroid_RAS.csv')
  node_names = list(parcel_info['ROI Name'])
  dummy_array = np.zeros((n_parcels,n_parcels))
  colnames = connection_names(dummy_array, node_names)
  uid_data = pd.DataFrame(np.load(f'{ucla_path}{uid}{sep}{uid}_FunctionalConnectomes.npy', allow_pickle=True), columns = np.load(f'{ucla_path}{uid}{sep}{uid}_FunctionalConnectomes_colnames.npy', allow_pickle=True))
  uid_data = uid_data[~uid_data[colnames].isna().any(axis=1)]
  scaled_uid_data = StandardScaler().fit_transform(uid_data[colnames])
  uid_plt_task = make_subplots(
    rows=len(n_list), 
    cols=len(d_list), 
    row_titles=n_list_str, 
    column_titles=d_list_str,
    x_title = 'Minimum Distance',
    y_title = 'N Neighbors'
  )
  for n in n_list:
    for d in d_list:
      try:
        umap_df = pd.read_csv(f'{ucla_path}{uid}{sep}{uid}_{n}-neighbors_d-{d}_UMAP.csv')
      except:
        fit = umap.UMAP(
          n_neighbors=n,
          min_dist=d,
          n_components=2,
          metric='euclidean'
        )
        umap_output = fit.fit_transform(scaled_uid_data)
        umap_df = pd.DataFrame(umap_output)
        umap_df.columns = ['x','y']
        umap_df['diagnosis'] = uid_data['diagnosis']
        umap_df['participant_id'] = uid_data['participant_id']
        umap_df['age'] = uid_data['age'].astype(int)
        umap_df['gender'] = uid_data['gender']
        umap_df['ScannerSerialNumber'] = uid_data['ScannerSerialNumber']
        umap_df['Task'] = uid_data['Task']
        umap_df = umap_df[~pd.isna(umap_df['Task'])]
        umap_df.to_csv(f'{ucla_path}{uid}{sep}{uid}_{n}-neighbors_d-{d}_UMAP.csv', index=False)
      scatter_task_1 = px.scatter(
        umap_df, 
        x='x',
        y='y',
        color='Task',
        opacity=1,
        size_max =.01,
        color_discrete_map =cmap_task,
        width=2000, 
        height = 1200
      )
      scatter_task_1.update_layout(
        showlegend=True,
        xaxis_title="X",
        yaxis_title="Y",
        legend_title="Task",
        font=dict(
          family="Times New Roman",
          size=30,
          color="Grey"
          ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
      )
      scatter_task_1.update_yaxes(
        showgrid=False,
        #gridcolor="Black",
        showline=False,
        linecolor="Grey",
        visible=True, showticklabels=True,
        zeroline=False
      )
      scatter_task_1.update_xaxes(
        showline=True,
        showgrid=False,
        linecolor="Grey",
        visible=True, showticklabels=True,
        zeroline=False
      )
      img_bytes = scatter_task_1.to_image(format="png", width=600, height=600, scale=2)
      Image(img_bytes)
      scatter_task_1.write_image(f'{ucla_path}{uid}{sep}{uid}_{n}-neighbors_d-{d}_task_UMAP.png')
      if d_list.index(d)==0 and n_list.index(n)==0:
        show_legend = True
      else:
        show_legend = False
      # task
      for dat in scatter_task_1.data:
        uid_plt_task.add_trace(
          go.Scatter(
            x=dat['x'],
            y=dat['y'],
            name=dat['name'],
            opacity=1,
            # color_discrete_map =cmap_task,
            legendgroup=dat['name'],
            showlegend=show_legend,
            marker={'color':cmap_task[dat['name']]},
            mode='markers'
          ),
          row=n_list.index(n)+1,
          col=d_list.index(d)+1,
        )
  aroma = uid_dict[uid]['AROMA']
  uid_plt_task.update_layout(
    height=2000, 
    width=2000, 
    title_text=f'Task UMAP {n_parcels} parcels, {aroma}',
    showlegend=True,
    font=dict(
      family="Times New Roman",
      # size=30,
      color="Grey"
      ),
    template='plotly_dark',
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)'
  )
  uid_plt_task.update_yaxes(
    showgrid=False,
    #gridcolor="Black",
    showline=True,
    linecolor="Grey",
    visible=True, 
    showticklabels=True,
    title={
      'font':dict(
        family="Times New Roman",
        # size=30,
        color="Grey"
      )
    },
    zeroline=False
  )
  uid_plt_task.update_xaxes(
    showline=True,
    showgrid=False,
    linecolor="Grey",
    visible=True, 
    showticklabels=True,
    title={
      'font':dict(
        family="Times New Roman",
        # size=30,
        color="Grey"
      )
    },
    zeroline=False
  )
  uid_plt_task.write_image(f'{ucla_path}{uid}{sep}{uid}_task_UMAP.png')
  uid_plt_task.write_html(f'{ucla_path}{uid}{sep}{uid}_task_UMAP.html')
  task_fig_list.append(uid_plt_task)

with open(f'{ucla_path}UMAP-TASK.html', 'a') as f:
  for fig in task_fig_list:
    f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))