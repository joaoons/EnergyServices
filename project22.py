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

except Exception as e:
    print("\U0001F525 Error during app startup:")
    print(traceback.format_exc())

    # Fallback layout to prevent crash on Render
    app.layout = html.Div([
        html.H1("⚠️ App failed to load"),
        html.P("Check the Render logs for details.")
    ])





