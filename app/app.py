from flask import Flask, flash, request, redirect, url_for, render_template, send_from_directory
from werkzeug.utils import secure_filename
from sklearn.preprocessing import OneHotEncoder
from datetime import datetime
from dateutil.relativedelta import relativedelta

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import pickle
import plotly
import plotly.express as px
import plotly.graph_objects as go
import json

upload_path_close = 'data/uploads/Close Order CM MT.csv'
upload_path_open = 'data/uploads/Open Order CM MT.csv'
upload_path = '../data/uploads'

app = Flask(__name__)
app.config.update(
	UPLOAD_PATH = upload_path
	)

def refresh_data():

	data_close_cm = pd.read_csv(upload_path_close)
	data_open_cm = pd.read_csv(upload_path_open)

	#-------------------------------
	# Procesando los datos de CLOSE
	#-------------------------------

	# Drop de los datos duplicados, que son los valores antes de pasuar el CM en el acknowledge
	data_close_cm = data_close_cm.drop_duplicates('cm_projectno', keep='last')

	# Creando una columna que identifique el TAT y si no tiene poner el TAT total
	data_close_cm['TAT_ack_to_close'] = data_close_cm.apply(lambda x: x.tat if x.TAT_ack_to_close > x.tat else x.TAT_ack_to_close, axis=1)

	# Se dropean los CM con TRF para que los valores medios no se disparen y en sucien el resultado.
	# Cuando se termine de limpiar se eliminará esta sección
	mask_trf = data_close_cm.orderno.str.contains('TRF', regex=False)
	data_close_cm = data_close_cm.drop(data_close_cm.loc[mask_trf].index)

	# Castear el formato de la fecha
	data_close_cm['order_date'] = pd.to_datetime(data_close_cm['order_date'])
	data_close_cm['year'] = data_close_cm['order_date'].dt.year
	data_close_cm['month'] = data_close_cm['order_date'].dt.month
	data_close_cm['day'] = data_close_cm['order_date'].dt.day
	data_close_cm['year_month'] = data_close_cm['order_date'].dt.strftime('%Y-%m')

	# Data wrangling (Limpieaza) de columnas que no sirven
	data_close_cm = data_close_cm.drop(['cm_projectno_i', 'detailno_i','orderno_i', 'detailno_i_rec', 'receiving_detail_number_id',
       'qty_rec', 'partno_rec', 'serial_batch_rec','DETAILNO_I_CM'], axis=1)

	# Diferencial de TAT entre el ACK y la creación del CM,  indica los tiempos de logística para alcanzar las herramientas al sector
	data_close_cm['TAT_logistic'] = data_close_cm['tat'] - data_close_cm['TAT_ack_to_close']

	# Reordenar las columnas
	data_close_cm = data_close_cm.loc[:,['cm_projectno', 'orderno', 'shop', 'partno', 'order_type', 'order_date',
       'closing_date', 'state', 'internal_repair', 'serial_batch_no',
       'detail_state', 'qty_order', 'backorder', 'description', 'condition',
       'receiving_number', 'delivery_date', 'condition_POST', 'ultima_act', 'ARRIVAL_IN_SHOP', 'PAUSED', 'PAUSING_REASON_I',
       'MUTATOR', 'year', 'month', 'day', 'year_month',
       'tat', 'TAT_logistic','TAT_ack_to_close']]

    # Crear un resumen DF para sacar los promedios y despues mergear con el data completo
	data_avg_month_lab = data_close_cm.groupby(pd.PeriodIndex(data_close_cm.order_date, freq="M"))['TAT_ack_to_close'].mean()
	data_avg_month_total = data_close_cm.groupby(pd.PeriodIndex(data_close_cm.order_date, freq="M"))['tat'].mean()
	data_avg_month_log = data_close_cm.groupby(pd.PeriodIndex(data_close_cm.order_date, freq="M"))['TAT_logistic'].mean()

	data_TAT = pd.concat([data_avg_month_total,data_avg_month_lab,data_avg_month_log], axis=1)

	data_TAT['year_month'] = data_TAT.index.to_timestamp()
	data_TAT['year_month'] = pd.to_datetime(data_TAT['year_month']).dt.strftime('%Y-%m')
	data_TAT = data_TAT.reset_index()

	data_TAT['indice'] = data_TAT.index.values

	data_TAT['TAT_KPI_month'] = data_TAT.indice.apply(lambda x: 0 if x == 0 else ((data_TAT.TAT_ack_to_close[x] - data_TAT.TAT_ack_to_close[x-1])/data_TAT.TAT_ack_to_close[x])*100)

	# Mergeo de data_TAT con el dataset completo
	data_close_cm = data_close_cm.merge(data_TAT, how='left', on='year_month', suffixes=['', '_mean'])

	#-------------------------------
	# Procesando los datos de OPEN
	#-------------------------------

	data_open_cm = data_open_cm.drop(columns='steps')
	data_open_cm = data_open_cm.drop(data_open_cm.loc[data_open_cm.SHOP == 'TOOL-MNT'].index)


	return data_close_cm, data_open_cm

@app.route('/')
@app.route('/home')
def index():

	data={
		'title': 'Main Dashboard',
		'area': 'Metrología & Laboratorio - v0.5'
	}

	data_close_cm, data_open_cm = refresh_data()

	print(data_close_cm)
	
	#---------------------------------------
	# INDICADORES
	#---------------------------------------
	indicadores = go.Figure()

	mask_open_paused = (data_open_cm.STATUS_CM == 'PAUSED - NO MANPOWER') | (data_open_cm.STATUS_CM == 'PAUSED - SENT TO EXTERNAL REPAIR') | (data_open_cm.STATUS_CM == 'PAUSED - AWAITING PHYSICAL DESTRUCTION')
	paused_cm_value = data_open_cm[mask_open_paused]

	mask_open_paused_other = data_open_cm.STATUS_CM == 'PAUSED - OTHER'
	paused_other_cm_value = data_open_cm[mask_open_paused_other]

	mask_close_TAT_mean = data_close_cm.order_date > '2021-12'
	tat_mean_total = data_close_cm[mask_close_TAT_mean].TAT_ack_to_close.mean()

	mask_close_TAT_mean_last_month = (data_close_cm.order_date < (datetime.now() - relativedelta(months=1)).strftime('%Y-%m')) & (data_close_cm.order_date > (datetime.now() - relativedelta(months=2)).strftime('%Y-%m'))
	tat_mean_last_month = data_close_cm[mask_close_TAT_mean_last_month].TAT_ack_to_close.mean()

	mask_close_TAT_mean_current_month = (data_close_cm.order_date < (datetime.now() - relativedelta(months=0)).strftime('%Y-%m')) & (data_close_cm.order_date > (datetime.now() - relativedelta(months=1)).strftime('%Y-%m'))
	tat_mean_current_month = data_close_cm[mask_close_TAT_mean_current_month].TAT_ack_to_close.mean()

	mask_running = data_open_cm.STATUS_CM == 'RUNNING'
	running_cm_value = data_open_cm[mask_running]

	mask_open = data_open_cm.STATUS_CM == 'OPEN'
	open_cm_value = data_open_cm[mask_open]

	mask_open_TRF = data_open_cm.STATUS_CM == 'OPEN-TRANSFER-TO-SHOP'
	open_TRF_cm_value = data_open_cm[mask_open_TRF]

	paused_shop_TAT_mean = data_open_cm.PAUSED_SHOP_TAT.mean()

	print(tat_mean_last_month)

	indicadores.add_trace(go.Indicator(
	    mode = "number+delta",
	    value = paused_shop_TAT_mean,
	    title = {"text": "TAT medio<br><span style='font-size:0.8em;color:gray'>Open</span>"},
	    domain = {'row': 0, 'column': 0}))

	indicadores.add_trace(go.Indicator(
	    mode = "number+delta",
	    value = running_cm_value.shape[0],
	    title = {"text": "RUNNING"},
	    domain = {'row': 1, 'column': 0}))

	indicadores.add_trace(go.Indicator(
	    mode = "number+delta",
	    value = paused_cm_value.shape[0],
	    title = {"text": "PAUSED"},
	    domain = {'row': 0, 'column': 1}))

	indicadores.add_trace(go.Indicator(
	    mode = "number+delta",
	    value = open_cm_value.shape[0],
	    title = {"text": "OPEN"},
	    domain = {'row': 1, 'column': 1}))

	indicadores.add_trace(go.Indicator(
	    mode = "number+delta",
	    value = paused_other_cm_value.shape[0],
	    title = {"text": "PAUSED - OTHER"},
	    domain = {'row': 0, 'column': 2}))

	indicadores.add_trace(go.Indicator(
	    mode = "number+delta",
	    value = open_TRF_cm_value.shape[0],
	    title = {"text": "OPEN - TRF"},
	    domain = {'row': 1, 'column': 2}))

	indicadores.add_trace(go.Indicator(
	    mode = "number+delta",
	    value = tat_mean_current_month,
	    title = {"text": "TAT medio<br><span style='font-size:0.8em;color:gray'>Último mes</span>"},
	    delta = {'reference': tat_mean_last_month, 'relative': True},
	    domain = {'row': 0, 'column': 3}))

	indicadores.add_trace(go.Indicator(
	    mode = "number+delta",
	    value = tat_mean_total,
	    title = {"text": "TAT medio<br><span style='font-size:0.8em;color:gray'>Total</span>"},
	    delta = {'reference': tat_mean_current_month, 'relative': False},
	    domain = {'row': 1, 'column': 3}))

	indicadores.update_layout(
    grid = {'rows': 2, 'columns': 4, 'pattern': "independent"})

	

	chart_indicadores = json.dumps(indicadores, cls = plotly.utils.PlotlyJSONEncoder)

	#---------------------------------------
	# GRAFICO DE TORTA
	#---------------------------------------
	labels_pie = data_open_cm.STATUS_CM.value_counts().index
	values_pie = data_open_cm.STATUS_CM.value_counts().values

	# Use `hole` to create a donut-like pie chart
	pie_fig = go.Figure(data=[go.Pie(labels=labels_pie, values=values_pie, hole=.3)])

	chart_pie = json.dumps(pie_fig, cls = plotly.utils.PlotlyJSONEncoder)
	

	#---------------------------------------
	# GRAFICO DE BARRAS TAT MEDIA
	#---------------------------------------


	years = [1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012]

	x=['b', 'a', 'c', 'd']

	fig_bar_1 = go.Figure(go.Bar(x=x, y=[2,5,1,9], name='Montreal'))
	fig_bar_1.add_trace(go.Bar(x=x, y=[1, 4, 9, 16], name='Ottawa'))
	fig_bar_1.add_trace(go.Bar(x=x, y=[6, 8, 4.5, 8], name='Toronto'))

	fig_bar_1.update_layout(barmode='stack',
							xaxis={'categoryorder':'category ascending'},
							width=800,
							height=375,
							margin=dict(l=5,
								        r=1,
								        b=20,
								        t=20,
								        pad=0)
					        )
	chart_bar_counts = json.dumps(fig_bar_1, cls = plotly.utils.PlotlyJSONEncoder)

	#---------------------------------------
	# GRAFICO DE BARRAS CANTIDAD DE LIBERACIONES AEP EZE
	#---------------------------------------
	mask_shop_AEP = (data_close_cm.shop == 'METROLOGY-A') & (data_close_cm.order_date > '2021')
	mask_shop_EZE = (data_close_cm.shop == 'METROLOGY') & (data_close_cm.order_date > '2021')

	data_close_cm_sorted = data_close_cm.sort_values(by=['order_date'], ascending=True)

	fig_bar_2 = go.Figure()
	fig_bar_2.add_trace(go.Bar(x=data_close_cm_sorted.year_month.value_counts().index.sort_values(ascending=True),
	                y=data_close_cm_sorted[mask_shop_AEP].year_month.value_counts().values,
	                name=data_close_cm_sorted[mask_shop_AEP].shop.value_counts().index[0],
	                marker_color='rgb(55, 83, 109)'
	                ))
	fig_bar_2.add_trace(go.Bar(x=data_close_cm_sorted.year_month.value_counts().index.sort_values(ascending=True),
	                y=data_close_cm_sorted[mask_shop_EZE].year_month.value_counts().values,
	                name=data_close_cm_sorted[mask_shop_EZE].shop.value_counts().index[0],
	                marker_color='rgb(26, 118, 255)'
	                ))

	fig_bar_2.update_layout(
	    title='US Export of Plastic Scrap',
	    xaxis_tickfont_size=14,
	    yaxis=dict(
	        title='USD (millions)',
	        titlefont_size=16,
	        tickfont_size=14,
	    ),
	    legend=dict(
	        x=0,
	        y=1.0,
	        bgcolor='rgba(255, 255, 255, 0)',
	        bordercolor='rgba(255, 255, 255, 0)'
	    ),
	    barmode='group',
	    bargap=0.15, # gap between bars of adjacent location coordinates.
	    bargroupgap=0.1, # gap between bars of the same location coordinate.
	    width=800,
		height=375,
		margin=dict(l=5,
			        r=5,
			        b=20,
			        t=30,
			        pad=10,
			        autoexpand=True)
	)
	fig_bar_2.update_xaxes(rangeslider_visible=True)
	
	chart_bar_TAT_mean = json.dumps(fig_bar_2, cls = plotly.utils.PlotlyJSONEncoder)

	return render_template('index.html', data=data, chart_bar_TAT_mean=chart_bar_TAT_mean, chart_bar_counts=chart_bar_counts, chart_indicadores=chart_indicadores, chart_pie=chart_pie)

def pagina_no_encontrada(error):

	data={
		'title':'Ups! Le pifiaste en algo. :(',
		'area': 'Metrología & Laboratorio'
	}

	return render_template('pagina_no_encontrada.html', data=data), 404

if __name__ == '__main__':
	app.register_error_handler(400, pagina_no_encontrada)
	app.run(debug=True, port=5000)