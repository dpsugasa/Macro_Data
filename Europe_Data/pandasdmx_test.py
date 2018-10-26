# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 10:02:37 2018

@author: dpsugasa
"""

from pandasdmx import Request
import plotly
import plotly.plotly as py #for plotting
import plotly.graph_objs as go
import plotly.dashboard_objs as dashboard
import plotly.tools as tls
import plotly.figure_factory as ff
import credentials #plotly API details

estat = Request('ESTAT')
flow_response = estat.dataflow('une_rt_a')
structure_response = flow_response.dataflow.une_rt_a.structure(request=True, target_only=False)
structure_response.write().codelist.loc['GEO'].head()

resp = estat.data('une_rt_a', key={'GEO': 'EL+ES+IT'}, params={'startPeriod': '1950'})

data = resp.write(s for s in resp.data.series if s.key.AGE == 'TOTAL')
z = data.loc[:, ('PC_ACT', 'TOTAL', 'T')]
z['Dates'] = z.index.to_series().astype(str)
z['IT_scr'] = (z['IT'] - z['IT'].mean())/z['IT'].std()
print(z)



# Create Probability Density by Strike
trace1 = go.Scatter(
            x = z['Dates'].values,
            y = z['IT'].values,
            xaxis = 'x1',
            #yaxis = 'y1',
            mode = 'lines+markers+text',
            line = dict(width=3, color= '#ffaf1a'),
            name = 'Unemployment',
            text = z['IT'].values,
            textposition = 'top left',
            textfont=dict(size=14),
            marker = dict(size=12),
            fill='tonexty',
            fillcolor = '#b3b3b3'
            )

        
    
layout  = {'title' : f'Italy Long-Term Unemployment Percentage',
                           'xaxis' : {'title' : 'Year',
                                      'fixedrange': True,
                                      'showgrid' : True},
                           'yaxis' : {'title' : 'Percentage',
                                      'fixedrange' : True,
                                      'showgrid' :True},
                            
        #                   'shapes': [{'type': 'rect',
        #                              'x0': d[i]['scr_1y'].index[0],
        #                              'y0': -2,
        #                              'x1': d[i]['scr_1y'].index[-1],
        #                              'y1': 2,
        #                              'name': 'Z-range',
        #                              'line': {
        #                                      'color': '#f48641',
        #                                      'width': 2,},
        #                                      'fillcolor': '#f4ad42',
        #                                      'opacity': 0.25,
        #                                      },]
                           }
data = [trace1]
figure = go.Figure(data = data, layout=layout)
py.iplot(figure, filename = f'Macro_Data/Italy_unemployment')
