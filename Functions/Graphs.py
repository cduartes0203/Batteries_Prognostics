import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import plot
import plotly.express as px
import numpy as np
import pandas as pd
from Functions.Utils import *

def plot_single(y,x=None,w=400,h=400,title='r'):
  if x is None: x = [i for i in range(len(y))]
  fig=make_subplots(rows=1,cols=1)
  trace = go.Scatter(x=x, y= y)
  fig.add_trace(trace, row=1, col=1)
  
  fig.update_layout(width = w, height = h, title = title)
  fig.update_yaxes( title_text='Amplitude', row = 1, col = 1)
  fig.show()

def plot_series(series, names=None, title='Séries Temporais',
                  markers=None, w=600, h=400, xrg=None, yrg=None,show=True):
    if xrg==None: xrg = [None, None]
    if yrg==None: yrg = [None, None]
    
    x=series[0]
    y=series[1]
    line_modes = ['lines', 'markers']
    fig = make_subplots(rows=1, cols=1)
    
    if names is None:
        names = [f'Série {i+1}' for i in range(len(y))]
    if markers is None:
        markers = [0] * len(y) # Padrão para todos como 'lines'

    for x, y, name, m_idx in zip(x, y, names, markers):
        mode = line_modes[m_idx] if m_idx < len(line_modes) else 'lines'
        fig.add_trace(go.Scatter(x=x, y=y, name=name, mode=mode),row=1, col=1)

    fig.update_layout(width=w, height=h, title=title,template='plotly_white')
    fig.update_yaxes(title_text='Amplitude', row=1, col=1,range=yrg)
    fig.update_xaxes(title_text='Tempo / Frequência', row=1, col=1,range=xrg)
    if show:
        fig.show()
    return fig

def PlotSeriesBySide(series_, w=900, h=400, title="Side-by-side", titles=None):
    n = len(series_)
    titles = titles if titles is not None else [f"Plot {i+1}" for i in range(n)]
    fig = make_subplots(rows=1, cols=n, subplot_titles=titles)

    for i, series in enumerate(series_, start=1):
        #print(len(series))
        f = plot_series(series, show=False)     # gera “mini-fig”
        for tr in f.data:
            fig.add_trace(tr, row=1, col=i) # reaproveita o trace
        fig.update_yaxes(title_text="Amplitude", row=1, col=i)

    fig.update_layout(width=w, height=h, title=title)
    fig.show()

def plot_2series(y1,y2,x1=None,x2=None,s1 ='series1',s2 ='series2',title='r', mrkr1 = 0, mrkr2 = 0, w=400,h=400,):
  if x1 is None: x1 = np.arange(len(y1))
  if x2 is None: x2 = np.arange(len(y2))
  line_modes = ['lines','markers'] 
  fig=make_subplots(rows=1,cols=1)
  fig.add_trace(go.Scatter(x=x1, y=y1, name=s1, mode = line_modes[mrkr1]), row=1, col=1)
  fig.add_trace(go.Scatter(x=x2, y=y2, name=s2,mode = line_modes[mrkr2]), row=1, col=1)
  fig.update_layout(width = w, height = h, title = title)
  fig.update_yaxes( title_text='Amplitude', row = 1, col = 1)
  fig.show()

def plot_single2(x, y, w=5, h=5, title='r'):

    plt.figure(figsize=(w, h))  # Define o tamanho do gráfico
    plt.plot(x, y, linestyle='-', marker='o', color='b', label="Sinal")  # Plota os dados
    plt.xlabel("Tempo")  # Nome do eixo X
    plt.ylabel("Amplitude")  # Nome do eixo Y
    plt.title(title)  # Define o título do gráfico
    plt.grid(True)  # Adiciona grade ao gráfico
    plt.legend()  # Exibe legenda
    plt.show()  # Mostra o gráfico

def plot_double(x1,y1,x2,y2,label1 = 'label1',label2 = 'label2', title='title'):
  fig=make_subplots(rows=1,cols=2)
  fig.add_trace(go.Scatter(x=x1, y= y1, name= label1), row=1, col=1)
  fig.add_trace(go.Scatter(x=x2, y= y2, name= label2), row=1, col=2)
  fig.update_layout(width = 800, height = 400, title = title)
  fig.update_yaxes( title_text='Amplitude', row = 1, col = 1)
  fig.show()

def plot_single_df(df,j=0):
  fig=make_subplots(rows=1,cols=1)
  trace = go.Scatter(x=[i for i in range(len(df))], y = df.iloc[:,j])
  fig.add_trace(trace, row=1, col=1)
  fig.update_layout(width = 400, height = 400, title = df.columns[j])
  fig.update_yaxes( title_text='Amplitude', row = 1, col = 1)
  fig.show()

def plot_3d(df,width=600,height=600,title = 'r',xlabel = 'Frequency (Hz)',ylabel = 'Sample',zlabel = 'Amplitude'):
  x = [[g  for g in range(len(df))] for i in range(len(df.columns))]
  y = []
  for i in range(len(df.columns)):
      y.append([i for j in range(len(df))])
  z = [df.iloc[:,i] for i in range(len(df.columns))]
  traces = []
  for i in range(len(x)):
      trace = go.Scatter3d(
          x=x[i],
          y=y[i],
          z=z[i],
          mode='lines',
          name=df.columns[i],
          line=dict(color=z[i], colorscale='Viridis', width=2),
      )
      traces.append(trace)

  layout = go.Layout(
      title=title,
      scene=dict(
          xaxis=dict(title=xlabel),
          yaxis=dict(title=ylabel),
          zaxis=dict(title=zlabel),
          camera=dict(
              eye=dict(x=-1.25, y=-1.25, z=0.75)
          )
      ),
      width=width, height = height
  )

  fig = go.Figure(data=traces, layout=layout)
  fig.show()
  return traces


def plot_3d_fft_2(df,width=600,height=600,title = 'r',xlabel = 'Frequency (Hz)',ylabel = 'Sample',zlabel = 'Amplitude'):
    x = [df.iloc[:, 0] for i in range(1, len(df.columns))]
    y = []
    for i in range(1, len(df.columns)):
        y.append([i for j in range(len(df))])
    z = [df.iloc[:, i] for i in range(1, len(df.columns))]
    
    traces = []
    for i in range(len(x)):
        trace = go.Scatter3d(
            x=x[i],
            y=y[i],
            z=z[i],
            mode='lines',
            name=df.columns[i + 1],
            line=dict(color=z[i], colorscale='Viridis', width=2),
        )
        traces.append(trace)
    
    return traces
def plot_3d_inline(fft_r,env_r,title1 = 'df1',title2 = 'df2'):

    fig = make_subplots( rows=1, cols=2, specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]], subplot_titles=(title1, title2),)

    fft_traces = plot_3d_fft_2(fft_r, 500, 500, 'fft_r', 'r', 'r', 'r')
    for trace in fft_traces:
        fig.add_trace(trace, row=1, col=1)

    env_traces = plot_3d_fft_2(env_r, 500, 500, 'env_r', 'r', 'r', 'r')
    for trace in env_traces:
        fig.add_trace(trace, row=1, col=2)

    fig.update_layout( height=600, width=1200, title_text="Comparison of FFT_R and ENV_R in 3D")
    fig.show()


def plot_custom(df_r,df_d):
    num_cols = len(df_r.columns[1:])

    fig, axes = plt.subplots(nrows=(num_cols + 2) // 3, ncols=3, figsize=(18, (num_cols + 2) // 3 * 3))

    axes = axes.flatten()

    for i in range(1, len(df_r.columns)):
        axes[i - 1].plot(df_r.iloc[:int(0.5*len(df_r)), i], label='df_r')
        axes[i - 1].plot(df_d.iloc[:int(0.5*len(df_r)), i], label='df_d')
        axes[i - 1].set_title(f'{df_r.columns[i]}')
        axes[i - 1].legend()

    if num_cols % 3 != 0:
        for j in range(3 - num_cols % 3):
            axes[-(j+1)].axis('off')

    plt.tight_layout()
    plt.show()



def plot_3d_fft(df,width=600,height=600,title = 'r',xlabel = 'Frequency (Hz)',ylabel = 'Sample',zlabel = 'Amplitude'):
    x = [df.iloc[:,0] for i in range(1,len(df.columns))]
    y = []
    for i in range(1,len(df.columns)):
        y.append([i for j in range(len(df))])
    z = [df.iloc[:,i] for i in range(1,len(df.columns))]
    traces = []
    for i in range(len(x)):
        trace = go.Scatter3d(
            x=x[i],
            y=y[i],
            z=z[i],
            mode='lines',
            name=df.columns[i+1],
            line=dict(color=z[i], colorscale='Viridis', width=2),
        )
        traces.append(trace)

    layout = go.Layout(
        title=title,
        scene=dict(
            xaxis=dict(title=xlabel),
            yaxis=dict(title=ylabel),
            zaxis=dict(title=zlabel),
            camera=dict(
                eye=dict(x=-1.25, y=-1.25, z=0.75)
            )
        ),
        width=width, height = height
    )

    fig = go.Figure(data=traces, layout=layout)
    fig.show()

def plot_dataframe2(df, df2, df3,label = '1',label1 = '2',label2 = '3'):
    # Number of columns and rows for subplots
    num_cols = len(df.columns)
    num_plots_per_row = 4
    num_rows = (num_cols + num_plots_per_row - 1) // num_plots_per_row  # Ceiling division

    # Create subplots with the given number of rows and columns (4 per row)
    fig, axes = plt.subplots(num_rows, num_plots_per_row, figsize=(20, 5 * num_rows))
    fig.suptitle("DataFrame Columns Plotted", fontsize=16)

    # Flatten axes array to easily iterate through (if needed for 2D grid of axes)
    axes = axes.flatten()

    # Generate x values (length of the DataFrame)
    x_values = list(range(len(df)))

    # Plot each column in a subplot
    for i, col in enumerate(df.columns):
        ax = axes[i]
        ax.plot(x_values, df[col], label=label, color='blue')
        ax.plot(x_values, df2[col], label=label1, color='orange')
        ax.plot(x_values, df3[col], label=label2, color='green')
        ax.set_title(col, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True)

    # Hide any unused subplots if the number of columns is not a multiple of 4
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for suptitle
    plt.show()


def plot_features(df, cols_qtd=4, brng='brng', show=True, w=12, h=2.5,ylim=[None,None],show_title=False):

    nc = cols_qtd  # Número de colunas
    nr = -(-df.shape[1] // nc)  # Cálculo do número de linhas (ceil)

    # Criar figura e subplots
    fig, axes = plt.subplots(nrows=nr, ncols=nc, figsize=(w, h))
    if show_title: fig.suptitle(f'Features - {brng}', fontsize=10)  # Título principal
    
    # Garantir que `axes` seja sempre uma matriz 2D
    if nr == 1 and nc == 1:
        axes = np.array([[axes]])  # Se for um único subplot, ajusta para matriz 2D
    elif nr == 1 or nc == 1:
        axes = np.reshape(axes, (-1, nc))  # Se for linha ou coluna única, ajusta matriz corretamente
    
    axes = axes.flatten()  # Achata a matriz para iteração fácil

    # Loop sobre cada coluna do DataFrame e adiciona ao subplot correspondente
    for i, column in enumerate(df.columns):
        axes[i].plot(df.index, df[column], label=f'{brng[:-4]}_'+column)
        axes[i].set_title(f'{brng}_'+column, fontsize=10)
        axes[i].set_ylim(ylim[0],ylim[1])
        axes[i].grid(True)
        axes[i].tick_params(axis="both", labelsize=8)

    # Remover subplots vazios, se existirem
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Ajustar layout para melhor visualização
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Evita sobreposição com título

    # Exibir o gráfico se necessário
    if show:
        plt.show()

def plot_features2(df,df2, cols_qtd=4, brng='brng', show=True, w=12, h=6,ylim=[None,None],show_title=False):

    nc = cols_qtd  # Número de colunas
    nr = -(-df.shape[1] // nc)  # Cálculo do número de linhas (ceil)

    # Criar figura e subplots
    fig, axes = plt.subplots(nrows=nr, ncols=nc, figsize=(w, h))
    if show_title: fig.suptitle(f'Features - {brng}', fontsize=10)  # Título principal
    
    # Garantir que `axes` seja sempre uma matriz 2D
    if nr == 1 and nc == 1:
        axes = np.array([[axes]])  # Se for um único subplot, ajusta para matriz 2D
    elif nr == 1 or nc == 1:
        axes = np.reshape(axes, (-1, nc))  # Se for linha ou coluna única, ajusta matriz corretamente
    
    axes = axes.flatten()  # Achata a matriz para iteração fácil

    # Loop sobre cada coluna do DataFrame e adiciona ao subplot correspondente
    for i, column in enumerate(df.columns):
        axes[i].plot(df.index, df[column], label=df.columns[i])
        axes[i].plot(df2.index, df2[df2.columns[i]], label=df2.columns[i])
        axes[i].set_title(column, fontsize=10)
        axes[i].set_ylim(ylim[0],ylim[1])
        axes[i].grid(True)
        axes[i].tick_params(axis="both", labelsize=8)

    # Remover subplots vazios, se existirem
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Ajustar layout para melhor visualização
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Evita sobreposição com título

    # Exibir o gráfico se necessário
    if show:
        plt.show()

def plot_featuresN(dfs, cols_qtd=4, brng='brng', show=True, w=12, h=2.5,ylim=[None,None],show_title=False):

    nc = cols_qtd  # Número de colunas
    nr = -(-dfs[0].shape[1] // nc)  # Cálculo do número de linhas (ceil)

    # Criar figura e subplots
    fig, axes = plt.subplots(nrows=nr, ncols=nc, figsize=(w, h))
    if show_title: fig.suptitle(f'Features - {brng}', fontsize=10)  # Título principal
    
    # Garantir que `axes` seja sempre uma matriz 2D
    if nr == 1 and nc == 1:
        axes = np.array([[axes]])  # Se for um único subplot, ajusta para matriz 2D
    elif nr == 1 or nc == 1:
        axes = np.reshape(axes, (-1, nc))  # Se for linha ou coluna única, ajusta matriz corretamente
    
    axes = axes.flatten()  # Achata a matriz para iteração fácil
    names = ['X','Y','Z']
    # Loop sobre cada coluna do DataFrame e adiciona ao subplot correspondente
    for i, column in enumerate(dfs[0].columns):
        for df,name in zip(dfs,names):
            axes[i].plot(df.index, df[df.columns[i]], label=name)

        axes[i].set_title(column, fontsize=10)
        axes[i].set_ylim(ylim[0],ylim[1])
        axes[i].grid(True)
        axes[i].tick_params(axis="both", labelsize=8)
        axes[i].legend(fontsize=5)

    # Remover subplots vazios, se existirem
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Ajustar layout para melhor visualização
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Evita sobreposição com título

    # Exibir o gráfico se necessário
    if show:
        plt.show()


def plot_metrics(df,out, cols_qtd, brng, show=True, w=12, h=6):
    """
    Plota métricas de Monotonicidade, Trendabilidade e Correlação no Matplotlib.

    Parâmetros:
    - df: DataFrame com os dados.
    - cols_qtd: Número de colunas de subplots.
    - brng: Nome do bearing (usado no título).
    - show: Se True, exibe o gráfico.
    - w, h: Dimensões da figura.
    
    Retorna:
    - df_r: DataFrame filtrado com colunas que atendem critérios (>0.5).
    """

    df_r = pd.DataFrame()
    categories = ['M', 'T', 'C']
    categories2 = ['Monotonicidade', 'Trendabilidade', 'Correlação']
    bar_colors = ['blue', 'orange', 'green']

    # 🔹 Determinar número de linhas e colunas
    cols_qtd = min(cols_qtd, df.shape[1])  # Limitar colunas ao máximo existente
    rows = -(-df.shape[1] // cols_qtd)  # Cálculo correto do número de linhas (ceil)

    # 🔹 Criar figura e eixos para subplots
    fig, axes = plt.subplots(nrows=rows, ncols=cols_qtd, figsize=(w, h))
    fig.suptitle(f'Rolamento{brng[7:-4]}: Métricas de Avaliação das Características',
                  fontsize=9, y =0.95)

    # 🔹 Achatar matriz de eixos para facilitar a iteração
    axes = np.array(axes).flatten()

    # 🔹 Loop sobre cada coluna do DataFrame
    for i, column in enumerate(df.columns):
        metrics = np.array([
            calculate_monotonicity(df[column].values),
            calculate_trendability(df[column].values),
            calculate_correlation(df[column].values)
        ])
        metrics = np.abs(metrics)

        # 🔹 Filtrar colunas com métricas > 0.5
        if metrics[0] > 0.5 and metrics[1] > 0.5:
            df_r[column] = df[column]

        # 🔹 Criar gráfico de barras
        axes[i].bar(categories, metrics, color=bar_colors, width=0.4,label = categories2)
        axes[i].set_ylim(0, 1)
        yticks = list(np.arange(0, 1.25, .25))
        axes[i].set_yticks(sorted(yticks))
        axes[i].set_yticklabels(sorted(yticks), color='black') 
        axes[i].set_title(column, fontsize=8)
        axes[i].tick_params(axis="x", labelrotation=0, labelsize=6)
        axes[i].tick_params(axis="y", labelrotation=0, labelsize=6)
        axes[i].grid(True, alpha=0.3, linestyle='--',linewidth=0.75,color='black')
        handles, labels = axes[i].get_legend_handles_labels() 
        if i == 12:
            axes[i].legend(handles[:1], labels[:1], fontsize=7, loc="lower center",bbox_to_anchor=(.5, -1))
        if i == 13:
            axes[i].legend(handles[1:2], labels[1:2], fontsize=7, loc="lower center",bbox_to_anchor=(.5, -1))
        if i == 14:
            axes[i].legend(handles[2:], labels[2:], fontsize=7, loc="lower center",bbox_to_anchor=(.5, -1))

    # 🔹 Remover subplots vazios
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # 🔹 Ajustar layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # 🔹 Exibir gráfico
    if show:
        plt.savefig(out+f'{brng[:-4]}_Metrics.eps', dpi=500)
        plt.show()
        plt.close(fig)  # Liberar memória

    return df_r

def plot_multiple_features(dfs,out, cols_qtd, brngs, labels, show=True, w=12, h=6):
    names = [brngs[i][7:-4] for i in range(len(brngs))]
    """
    Plota múltiplas séries temporais de até três DataFrames em um conjunto de subplots.

    Parâmetros:
    - dfs: Lista contendo os DataFrames a serem comparados (exatamente 3).
    - cols_qtd: Número de colunas de subplots.
    - brngs: Lista com os nomes dos bearings (usado no título).
    - labels: Lista com rótulos para os DataFrames na legenda.
    - show: Se True, exibe o gráfico.
    - w: Largura da figura.
    - h: Altura da figura.

    Retorna:
    - None (exibe o gráfico).
    """

    # Pegar as colunas do primeiro DataFrame (assume que os 3 têm as mesmas colunas)
    columns = dfs[0].columns
    nc = cols_qtd  # Número de colunas de subplots
    nr = -(-len(columns) // nc)  # Cálculo correto do número de linhas (ceil)

    # Criar figura e subplots
    fig, axes = plt.subplots(nrows=nr, ncols=nc, figsize=(w, h))
    fig.suptitle(f'Comparação de Características - Rolamentos{", ".join(names[:-1])} e {names[-1]} '
                 , fontsize=9, y =0.95)


    # Garantir que `axes` seja uma matriz 2D
    axes = np.array(axes).flatten()

    # 🔹 Loop sobre cada coluna para plotar os três DataFrames no mesmo subplot
    for i, column in enumerate(columns):
        for df, label in zip(dfs, labels):
            #print(label)
            axes[i].plot(df.index, df[column], label=label)
        #print(column)
        axes[i].set_title(column, fontsize=8)
        axes[i].grid(True)
        axes[i].tick_params(axis="both", labelsize=10)
        axes[i].legend(fontsize=5,loc="center right")
        axes[i].set_xlabel("Ciclo", fontsize=5) 
        axes[i].set_ylabel("Magnitude", fontsize=5) 
        #axes[i].set_xlim(0, 123)

    # Remover subplots vazios, se existirem
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Ajustar layout para melhor visualização
    plt.tight_layout(rect=[0, 0, 1, 0.96])


    # Exibir o gráfico se necessário
    if show:
        plt.savefig(out+'Features.eps', dpi=500)
        plt.savefig(out+'Features.png', dpi=500)
        plt.show()