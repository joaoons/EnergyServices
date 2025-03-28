from dash import Dash, dcc, html, dash_table
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
from sklearn import metrics
import joblib 
import plotly.express as px
import traceback

# Initialize Dash
app = Dash(__name__)
server = app.server

try:
    # Load CSV Data
    df2 = pd.read_csv("IST_North_Tower_2019_raw.csv")  
    df1 = pd.read_csv("IST_North_Tower_Clean.csv")  
    dft = pd.read_csv("IST_North_Tower_test.csv")  

    # Prepare Data
    df2.rename(columns={'North Tower (kWh)': 'Power_kW'}, inplace=True)
    df1['Unnamed: 0'] = pd.to_datetime(df1['Unnamed: 0'])
    df1.set_index('Unnamed: 0', inplace=True)
    df2['Date'] = pd.to_datetime(df2['Date'], errors='coerce')
    df2.set_index('Date', inplace=True)
    df = pd.concat([df1, df2])
    df['Year'] = df.index.year
    available_years = sorted(df['Year'].dropna().unique())
    dft['Date'] = pd.to_datetime(dft['Date'], errors='coerce')
    dft.set_index('Date', inplace=True)

    # Prepare Forecast data
    Z = dft.values
    Y = Z[:, 0]
    X = Z[:, [1, 2, 3, 6, 5, 4]] 

    # Load models
    XGB_model = joblib.load('XGB_model.sav')
    y_pred_XGB = XGB_model.predict(X)

    NN_model = joblib.load('NN_model.sav')
    y_pred_NN = NN_model.predict(X)

    BT_model = joblib.load('BT_model.sav')
    y_pred_BT = BT_model.predict(X)

    sensor_titles = {
        'temp_C': 'Temperature', 'HR': 'Humidity', 'windSpeed_m/s': 'Wind speed',
        'windGust_m/s': 'Wind gust', 'pres_mbar': 'Pressure', 'solarRad_W/m2': 'Solar radiance',
        'rain_mm/h': 'Rain', 'rain_day': 'Rain'
    }

    sensor_units = {
        'temp_C': 'T (ºC)', 'HR': 'HR (%)', 'windSpeed_m/s': 'WG (m/s)',
        'windGust_m/s': 'WS (m/s)', 'pres_mbar': 'p (hPa)', 'solarRad_W/m2': 'SR (W/m2)',
        'rain_mm/h': 'Rain (mm/h)', 'rain_day': 'Rain'
    }

    app.layout = html.Div(children=[
        html.Div([
            html.H1('IST North Tower Energy Monitor', className='dashboard-title'),
            html.Img(src='/assets/logo.png', className='dashboard-logo')
        ], className='header'),

        dcc.Tabs([
            dcc.Tab(label='Real data', children=[
                html.H3("Yearly Energy Consumption", className='section-title'),
                dcc.Graph(id='bar-chart'),
                html.Div('Select year(s):', style={'padding-bottom': '10px'}),
                html.Div([
                    dcc.Dropdown(
                        id='menu',
                        options=[{'label': y, 'value': y} for y in available_years],
                        value=[available_years[2]],
                        multi=True
                    )
                ], style={'width': '30%', 'padding': '10px'}),
                html.H4("Energy Consumption", className='section-title'),
                dcc.Graph(id='time-series-graph'),
                html.H5("Weather data", className='section-title'),
                html.Div([
                    html.Div([
                        html.Label('Select weather variables:'),
                        dcc.Dropdown(
                            id='weather-variable-selector',
                            options=[{'label': sensor_titles.get(col, col), 'value': col} for col in sensor_titles.keys()],
                            value=list(sensor_titles.keys())[:2],
                            multi=True
                        )
                    ], style={'width': '48%', 'padding': '10px'}),
                    html.Div([
                        html.Label('Select chart type:'),
                        dcc.Dropdown(
                            id='weather-chart-type',
                            options=[
                                {'label': 'Line', 'value': 'line'},
                                {'label': 'Box', 'value': 'box'},
                                {'label': 'Histogram', 'value': 'histogram'}
                            ],
                            value='line'
                        )
                    ], style={'width': '48%', 'padding': '10px'})
                ], style={'display': 'flex', 'justifyContent': 'space-between'}),
                html.Div(id='sensor-plots')
            ]),
            dcc.Tab(label='Forecast', children=[
                html.H6("Forecast 2019 Power consumption", className='section-title'),
                html.Div([
                    html.Label('Select models:'),
                    dcc.Dropdown(
                        id='model-selector',
                        options=[
                            {'label': 'XGBoost', 'value': 'XGBoost'},
                            {'label': 'Neural Networks', 'value': 'Neural Networks'},
                            {'label': 'Boostrapping', 'value': 'Boostrapping'}
                        ],
                        value=['XGBoost', 'Boostrapping'],
                        multi=True
                    )
                ], style={'width': '40%', 'padding': '10px'}),
                html.Div([
                    html.Label('Select metrics:'),
                    dcc.Dropdown(
                        id='metric-selector',
                        options=[
                            {'label': 'MAE', 'value': 'MAE'},
                            {'label': 'MBE', 'value': 'MBE'},
                            {'label': 'MSE', 'value': 'MSE'},
                            {'label': 'RMSE', 'value': 'RMSE'},
                            {'label': 'cvRMSE', 'value': 'cvRMSE'},
                            {'label': 'NMBE', 'value': 'NMBE'}
                        ],
                        value=['MAE', 'RMSE'],
                        multi=True
                    )
                ], style={'width': '40%', 'padding': '10px'}),
                dcc.Graph(id='forecast-graph'),
                html.Div(id='forecast-info')
            ])
        ]),
        html.Footer("João Santos | Energy Services | March 2025", className='footer')
    ])

    @app.callback(Output('time-series-graph', 'figure'), Input('menu', 'value'))
    def update_time_series(selected_years):
        traces = []
        for year in selected_years:
            filtered_df = df[df['Year'] == year]
            traces.append({
                'x': filtered_df.index,
                'y': filtered_df['Power_kW'],
                'type': 'line',
                'name': f'Power - {year}'
            })
        return {
            'data': traces,
            'layout': {
                'title': 'Power Usage Over Time',
                'xaxis': {'title': 'Date'},
                'yaxis': {'title': 'P (kW)'},
                'legend': {'title': 'Legend'}
            }
        }

    @app.callback(Output('bar-chart', 'figure'), Input('menu', 'value'))
    def update_bar_chart(_):
        bar_data = df.groupby('Year')['Power_kW'].sum().loc[[2017, 2018, 2019]] / 1e6
        return {
            'data': [{
                'x': [str(year) for year in bar_data.index],
                'y': bar_data.values,
                'type': 'bar',
                'name': 'Yearly Total Power'
            }],
            'layout': {
                'title': 'Total Yearly Electricity Consumption (GWh)',
                'xaxis': {'title': 'Year', 'type': 'category'},
                'yaxis': {'title': {'text': 'P (GW)'}},
                'legend': {'title': 'Legend'}
            }
        }

    @app.callback(
        Output('sensor-plots', 'children'),
        Input('menu', 'value'),
        Input('weather-variable-selector', 'value'),
        Input('weather-chart-type', 'value')
    )
    def update_sensor_plots(selected_years, selected_vars, chart_type):
        filtered_df = df[df['Year'].isin(selected_years)]
        children = []
        for i in range(0, len(selected_vars), 2):
            row = []
            for col in selected_vars[i:i+2]:
                if chart_type == 'line':
                    data = [{
                        'x': filtered_df.index,
                        'y': filtered_df[col],
                        'type': 'line',
                        'name': sensor_titles.get(col, col)
                    }]
                    layout = {
                        'title': {'text': sensor_titles.get(col, col)},
                        'xaxis': {'title': 'Date'},
                        'yaxis': {'title': {'text': sensor_units.get(col, '')}},
                        'margin': {'l': 40, 'r': 10, 't': 30, 'b': 40}
                    }
                elif chart_type == 'box':
                    data = [{
                        'y': filtered_df[col],
                        'type': 'box',
                        'name': sensor_titles.get(col, col)
                    }]
                    layout = {
                        'title': {'text': f"{sensor_titles.get(col, col)}"},
                        'yaxis': {'title': {'text': sensor_units.get(col, '')}},
                        'margin': {'l': 40, 'r': 10, 't': 30, 'b': 40}
                    }
                elif chart_type == 'histogram':
                    data = [{
                        'x': filtered_df[col],
                        'type': 'histogram',
                        'name': sensor_titles.get(col, col)
                    }]
                    layout = {
                        'title': {'text': f"{sensor_titles.get(col, col)}"},
                        'xaxis': {'title': {'text': sensor_units.get(col, '')}},
                        'yaxis': {'title': 'Count'},
                        'margin': {'l': 40, 'r': 10, 't': 30, 'b': 40}
                    }
                else:
                    data = []
                    layout = {}
                fig = {'data': data, 'layout': layout}
                row.append(html.Div([
                    dcc.Graph(figure=fig, style={'height': '300px'})
                ], style={'width': '48%', 'margin': '1%'}))
            children.append(html.Div(row, style={'display': 'flex', 'justifyContent': 'space-between', 'flexWrap': 'wrap'}))
        return children

    @app.callback(
        Output('forecast-graph', 'figure'),
        Output('forecast-info', 'children'),
        Input('model-selector', 'value'),
        Input('metric-selector', 'value')
    )
    def update_forecast(selected_models, selected_metrics):
        dates = dft.index
        model_preds = {
            'XGBoost': y_pred_XGB,
            'Neural Networks': y_pred_NN,
            'Boostrapping': y_pred_BT
        }
        df_plot = pd.DataFrame({
            'Date': dates,
            'Power (kW)': Y,
            'Type': 'Real'
        })
        for model in selected_models:
            df_plot = pd.concat([df_plot, pd.DataFrame({
                'Date': dates,
                'Power (kW)': model_preds[model],
                'Type': model
            })])
        fig = px.line(df_plot, x='Date', y='Power (kW)', color='Type',
                      title='Forecasted vs Real Power Consumption')
        fig.update_layout(
            xaxis_title='Date',
            yaxis_title='Power (kW)',
            margin={'l': 60, 'r': 20, 't': 40, 'b': 40},
            legend_title='Model'
        )
        def get_metrics(name, y_pred):
            values = {
                'Model': name,
                'MAE': f'{metrics.mean_absolute_error(Y, y_pred):.2f}',
                'MBE': f'{np.mean(Y - y_pred):.2f}',
                'MSE': f'{metrics.mean_squared_error(Y, y_pred):.2f}',
                'RMSE': f'{np.sqrt(metrics.mean_squared_error(Y, y_pred)):.2f}',
                'cvRMSE': f'{(np.sqrt(metrics.mean_squared_error(Y, y_pred)) / np.mean(Y)):.2%}',
                'NMBE': f'{(np.mean(Y - y_pred) / np.mean(Y)):.2%}'
            }
            return values
        metrics_data = [get_metrics(name, pred) for name, pred in model_preds.items() if name in selected_models]
        table_columns = ['Model'] + selected_metrics
        metrics_table = dash_table.DataTable(
            columns=[{'name': col, 'id': col} for col in table_columns],
            data=[{col: row[col] for col in table_columns} for row in metrics_data],
            style_cell={
                'textAlign': 'center', 'padding': '4px', 'fontSize': '14px',
                'minWidth': '80px', 'maxWidth': '120px', 'width': '100px'
            },
            style_header={
                'fontWeight': 'bold', 'backgroundColor': '#f9f9f9'
            },
            style_table={'overflowX': 'auto', 'width': '60%', 'margin': 'auto'}
        )
        return fig, metrics_table

except Exception as e:
    print("🔥 Error during app startup:")
    print(traceback.format_exc())

