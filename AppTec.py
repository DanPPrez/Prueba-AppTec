import dash
import dash_core_components as dcc
import dash_html_components as html
import csv
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv
from dash.dependencies import Input, Output
from numpy import genfromtxt
                    ### Creacion de la app ###
app = dash.Dash()


                    ### Leer archivo csv ###
Datos = genfromtxt('Array.csv', delimiter=',')

                    ### Codigo de Minimos Cuadrados ###
x = Datos[:,0]
y = Datos[:,1]
#Parametros Iniciales
n = len(x); sg = 1; ajuste = 1; xn = x; yn = y;
#Determina el grado del polinomio
m = 3
# Obtencion Matriz sx
sx = np.empty((m+2,m+2))
for i in range(1,m+2):
	for j in range(1,m+2):
		sx[i][j] = sum(pow(xn,(i+j-2)))
sx = np.delete(sx,(0), axis=0); sx = np.delete(sx,(0), axis=1)
# Obtencion Vector sy
sy = np.empty((m+2,1))
for i in range(1,m+2):
    ml = pow(xn,i-1) 
    sy[i] = sum(yn*ml)
sy = np.delete(sy,(0), axis=0);
isx = inv(sx)
                    ### Ecuacion ###
c = isx.dot(sy)
txt = np.empty((m+1,m+1))
for w in range(0,m+1):
    txt[w] = print((c[w]),w)
#Valores para graficar
xx = np.linspace(min(x), max(x),100)
cn = np.flipud(c)
ya = np.polyval(cn,x)
yy = np.polyval(cn,xx)
                    ###  Grafica ###
app.layout = html.Div(children=[
    html.H1(children='Prueba AppTec',style={'textAlign': 'center'}),
    dcc.Graph(
        id='example',
        figure={
            'data': [
            {'x': x, 'y': y, 'name': 'Puntos', 'mode': 'markers', 'marker': {'size': 10}},
            {'x': xx, 'y': yy, 'type': 'line', 'name': 'Curva'}],
            'layout': {'title': 'Gráfica', 'xaxis':{'title':'Eje X'}, 'yaxis':{'title':'Eje Y'}}       
            })
    ])

if __name__ == '__main__':
    app.run_server(debug=True)