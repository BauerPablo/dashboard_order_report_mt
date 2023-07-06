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
	# Procesamiento de los datos de CLOSE
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
	# Procesamiento de los datos de OPEN
	#-------------------------------

	data_open_cm = data_open_cm.drop(columns='steps')
	data_open_cm = data_open_cm.drop(data_open_cm.loc[data_open_cm.SHOP == 'TOOL-MNT'].index)


	return data_close_cm, data_open_cm

@app.route('/')
@app.route('/home')
def index():
	# Esta función retorna un render_template en HTML y las variables de los graficos en JSON

	# Información de la pagina
	data={
		'title': 'Main Dashboard',
		'area': 'Metrología & Laboratorio - v0.5'
	}

	# Importando los datos de la BBDD
	data_close_cm, data_open_cm = refresh_data()
	
	#---------------------------------------
	# INDICADORES
	# Labels que indican KPI's de gestión del laboratorio
	# - TAT promedios
	# - TAT promedios del ultimo mes y diferencia con el anterior
	# - TAT de ordenes abiertas
	# - Cantidad de unidades en cada uno de los estados (OPEN, RUNNING, PAUSED, etc.)
	#
	# TO-DO: desplegable para diferenciar entre AEP y EZE
	#---------------------------------------

	# Instancia de los indicadores
	indicadores = go.Figure()

	# Ordenes Pausadas. Agrupadas entre NO MANPOWER, SENT TO EXTERNAL y SCRAP
	mask_open_paused = (data_open_cm.STATUS_CM == 'PAUSED - NO MANPOWER') | (data_open_cm.STATUS_CM == 'PAUSED - SENT TO EXTERNAL REPAIR') | (data_open_cm.STATUS_CM == 'PAUSED - AWAITING PHYSICAL DESTRUCTION')
	paused_cm_value = data_open_cm[mask_open_paused]

	# Ordenes Pausadas. Solo OTHERS
	mask_open_paused_other = data_open_cm.STATUS_CM == 'PAUSED - OTHER'
	paused_other_cm_value = data_open_cm[mask_open_paused_other]

	# TAT promedio total historico
	mask_close_TAT_mean = data_close_cm.order_date > '2021-12'
	tat_mean_total = data_close_cm[mask_close_TAT_mean].TAT_ack_to_close.mean()

	# TAT promedio ultimo mes
	mask_close_TAT_mean_last_month = (data_close_cm.order_date < (datetime.now() - relativedelta(months=1)).strftime('%Y-%m')) & (data_close_cm.order_date > (datetime.now() - relativedelta(months=2)).strftime('%Y-%m'))
	tat_mean_last_month = data_close_cm[mask_close_TAT_mean_last_month].TAT_ack_to_close.mean()

	# TAT promedio mes actual
	mask_close_TAT_mean_current_month = (data_close_cm.order_date < (datetime.now() - relativedelta(months=0)).strftime('%Y-%m')) & (data_close_cm.order_date > (datetime.now() - relativedelta(months=1)).strftime('%Y-%m'))
	tat_mean_current_month = data_close_cm[mask_close_TAT_mean_current_month].TAT_ack_to_close.mean()

	# Ordenes RUNNING
	mask_running = data_open_cm.STATUS_CM == 'RUNNING'
	running_cm_value = data_open_cm[mask_running]

	# Ordenes OPEN
	mask_open = data_open_cm.STATUS_CM == 'OPEN'
	open_cm_value = data_open_cm[mask_open]

	# Ordenes TRF
	mask_open_TRF = data_open_cm.STATUS_CM == 'OPEN-TRANSFER-TO-SHOP'
	open_TRF_cm_value = data_open_cm[mask_open_TRF]

	# TAT medio de las ordenes en estado de PAUSA
	paused_shop_TAT_mean = data_open_cm.PAUSED_SHOP_TAT.mean()

	print(tat_mean_last_month)

	# Indicadores. Se adjuntan a necesidad en una grilla de 2x4
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

	# configuración de la grilla
	indicadores.update_layout(
    grid = {'rows': 2, 'columns': 4, 'pattern': "independent"})

	# Creación de JSON de los indicadores para llevarlos a HTML
	chart_indicadores = json.dumps(indicadores, cls = plotly.utils.PlotlyJSONEncoder)

	#---------------------------------------
	# GRAFICO DE TORTA
	# Muestra la proporción de Orders abiertas, categorizada por el STATUS
	#---------------------------------------

	# Se genera una máscara para el grafico de torta y el resto de graficos para limitar los valores de la base de datos,
	# despues del confinamiento severo, a partir del dec 2020.
	mask_mayor_2021 = (data_close_cm.order_date > '2021')

	# Se traen los datos y se crea un DF especifico para el gráfico
	data_close_cm_STATUS_fig = pd.DataFrame(data_open_cm[mask_mayor_2021][['STATUS_CM']].value_counts()).reset_index()

	# Use `hole` to create a donut-like pie chart
	# Grafico de torta
	pie_fig = px.pie(data_close_cm_STATUS_fig,
						values=0,
						names='STATUS_CM',
						color_discrete_sequence= px.colors.sequential.Teal_r,
						hole=.3)

	pie_fig.update_layout(
		)
	#pie_fig = go.Figure(data=[go.Pie(labels=labels_pie, values=values_pie, hole=.3)])

	# Creación de JSON para llevarlos a HTML
	chart_pie = json.dumps(pie_fig, cls = plotly.utils.PlotlyJSONEncoder)
	

	#---------------------------------------
	# GRAFICO DE BARRAS TAT MEDIA
	# Indica el TAT medio de cada mes y compara con el mismo mes de los dos años anteriores
	# Por alguna razon no me deja cambiar los colores. No se por qué.
	#---------------------------------------

	# DF para crear los valores del grafico
	data_close_cm_TAT_fig = pd.DataFrame(data_close_cm[mask_mayor_2021][['year', 'month', 'TAT_ack_to_close_mean']].value_counts()).reset_index().sort_values(by=['year'])

	# Grafico de barras
	fig_bar_1 = px.bar(data_close_cm_TAT_fig,
						x='month',
						y='TAT_ack_to_close_mean',
						color='year',
						color_discrete_sequence= px.colors.sequential.Teal,
						text_auto=True)

	fig_bar_1.update_layout(title='Comparativa TAT [año a año]',
							barmode='group',
							bargap=0.15, # gap between bars of adjacent location coordinates.
						    bargroupgap=0.1, # gap between bars of the same location coordinate.
							width=800,
							height=375,
							margin=dict(l=5,
								        r=1,
								        b=20,
								        t=30,
								        pad=0)							
							)

	# Barra de desplazamiento por las fechas
	fig_bar_1.update_xaxes(rangeslider_visible=True)

	# Creación de JSON para llevarlos a HTML
	chart_bar_counts = json.dumps(fig_bar_1, cls = plotly.utils.PlotlyJSONEncoder)

	#---------------------------------------
	# GRAFICO DE BARRAS CANTIDAD DE LIBERACIONES AEP EZE
	# Grafico de barras que indica la cantidad de liberaciones mes a mes de los dos laboratorios AEP y EZE
	#---------------------------------------

	data_close_cm_fig = pd.DataFrame(data_close_cm[mask_mayor_2021][['year_month','shop']].value_counts()).reset_index()

	fig_bar_2 = px.bar(data_close_cm_fig,
						x='year_month',
						y=0,
						color='shop',
						color_discrete_map={
							'METROLOGY': 'rgb(55, 83, 109)',
							'METROLOGY-A': 'rgb(26, 118, 255)'},
						text_auto=True)

	fig_bar_2.update_layout(
	    title='Liberaciones AEP y EZE',
	    xaxis_tickfont_size=14,
	    xaxis=dict(
	    	title='Meses'
	    ),
	    yaxis=dict(
	        title='Cantidad de unidades',
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

	# Barra de desplazamiento por las fechas
	fig_bar_2.update_xaxes(rangeslider_visible=True)
	
	# Creación de JSON para llevarlos a HTML
	chart_bar_TAT_mean = json.dumps(fig_bar_2, cls = plotly.utils.PlotlyJSONEncoder)

	return render_template('index.html', data=data, chart_bar_TAT_mean=chart_bar_TAT_mean, chart_bar_counts=chart_bar_counts, chart_indicadores=chart_indicadores, chart_pie=chart_pie)

def pagina_no_encontrada(error):

	# Página no encontrada y su mensaje de error
	data={
		'title':'Ups! Le pifiaste en algo. :(',
		'area': 'Metrología & Laboratorio'
	}

	return render_template('pagina_no_encontrada.html', data=data), 404

if __name__ == '__main__':
	app.register_error_handler(400, pagina_no_encontrada)
	app.run(debug=True, port=5000)
	#TO-DO: hacer Tabs con otros graficos relevantes a medida que los soliciten.