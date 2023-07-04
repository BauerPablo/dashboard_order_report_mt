from flask import Flask, flash, request, redirect, url_for, render_template, send_from_directory
from werkzeug.utils import secure_filename
from sklearn.preprocessing import OneHotEncoder

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

upload_path = '../data/uploads'

app = Flask(__name__)
app.config.update(
	UPLOAD_PATH = upload_path
	)

@app.route('/')
@app.route('/home')
def index():

	data={
		'title': 'Main Dashboard',
		'area': 'Metrología & Laboratorio'
	}

	years = [1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012]

	x=['b', 'a', 'c', 'd']

	fig_bar_1 = go.Figure(go.Bar(x=x, y=[2,5,1,9], name='Montreal'))
	fig_bar_1.add_trace(go.Bar(x=x, y=[1, 4, 9, 16], name='Ottawa'))
	fig_bar_1.add_trace(go.Bar(x=x, y=[6, 8, 4.5, 8], name='Toronto'))

	fig_bar_1.update_layout(barmode='stack', xaxis={'categoryorder':'category ascending'})

	fig_bar_2 = go.Figure()
	fig_bar_2.add_trace(go.Bar(x=years,
	                y=[219, 146, 112, 127, 124, 180, 236, 207, 236, 263,
	                   350, 430, 474, 526, 488, 537, 500, 439],
	                name='Rest of world',
	                marker_color='rgb(55, 83, 109)'
	                ))
	fig_bar_2.add_trace(go.Bar(x=years,
	                y=[16, 13, 10, 11, 28, 37, 43, 55, 56, 88, 105, 156, 270,
	                   299, 340, 403, 549, 499],
	                name='China',
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
	    bargroupgap=0.1 # gap between bars of the same location coordinate.
	)

	chart_bar_counts = json.dumps(fig_bar_1, cls = plotly.utils.PlotlyJSONEncoder)
	chart_bar_TAT_mean = json.dumps(fig_bar_2, cls = plotly.utils.PlotlyJSONEncoder)

	return render_template('index.html', data=data, chart_bar_TAT_mean=chart_bar_TAT_mean, chart_bar_counts=chart_bar_counts)

def pagina_no_encontrada(error):

	data={
		'title':'Ups! Le pifiaste en algo. :(',
		'area': 'Metrología & Laboratorio'
	}

	return render_template('pagina_no_encontrada.html', data=data), 404

if __name__ == '__main__':
	app.register_error_handler(400, pagina_no_encontrada)
	app.run(debug=True, port=5000)