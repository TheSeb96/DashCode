import dash_core_components as dcc
import dash_html_components as html
import dash_table
import dash
import pandas as pd
from dash.dependencies import Input, Output, State
import numpy as np
import scipy.stats as si
from scipy.stats import norm

app = dash.Dash()

#Titre de la page web sur l'onglet
app.title = 'Interface for Risks gestion'

#importation du CSS
app.css.append_css({'external_url': 'https://codepen.io/amyoshino/pen/jzXypZ.css'})

#jeu de donnee
donneeTest = pd.read_csv("PI2_CAC40_FTSE100_DAX(sans_tickers).csv")

#stockage des données dans une dataframe. 
donneeStockagePortefeuille = pd.DataFrame(columns = ['Name', 'Type', 'Volume', 'Volatility', 'Strike', 'Spot', 'Interest rate', 'Maturity', 'Dividend', 'Price', 'Delta', 'Gamma', 'Theta', 'Rho', 'Vega'])
donneeTotalGreeks1 = pd.DataFrame(columns = ['Variation', 'Delta', 'Gamma', 'Theta', 'Rho', 'Vega'])

#fonction de calcul du prix de l'option (call/put)
def euro_vanilla_dividend(S, K, T, r, q, sigma, option, volume):
    #S: spot price
    #K: strike price
    #T: time to maturity
    #r: interest rate
    #q: rate of continuous dividend paying asset 
    #sigma: volatility of underlying asset
    #option : Call or Put
    r = r/100
    sigma = sigma/100
    q = q / 100
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - q - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    result = 0
    if option == 'Call':
        result = (S * np.exp(-q * T) * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0))
    if option == 'Put':
        result = (K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0) - S * np.exp(-q * T) * si.norm.cdf(-d1, 0.0, 1.0))
    result = result * volume
    result = round (result, 3)
    return result

#Fonction de calculs de Greeks

def deltaCall(S,K,r,q,T,sigma,volume):
    r = r/100
    sigma = sigma/100
    q = q / 100
    valeur = np.exp(-q*(T))*norm.cdf((1/sigma*np.sqrt(T))*(np.log(S/K)+(r-q+0.5*sigma*sigma)*(T)),0,1) 
    valeur = valeur * volume
    delta = round (valeur, 3)
    return delta

def deltaPut(S,K,r,q,T,sigma,volume):
    r = r/100
    sigma = sigma/100
    q = q / 100
    valeur = np.exp(-q*(T))*(norm.cdf((1/sigma*np.sqrt(T))*(np.log(S/K)+(r-q+0.5*sigma*sigma)*(T)),0,1)-1)
    valeur = valeur * volume
    delta = round (valeur, 3)
    return delta

def gamma(S,K,r,q,T,sigma,volume):
    r = r/100
    sigma = sigma/100
    q = q / 100
    valeur = np.exp(-q*(T))*(1/S*sigma*np.sqrt(T))*norm.pdf((1/sigma*np.sqrt(T))*(np.log(S/K)+(r-q+0.5*sigma*sigma)*(T)),0,1)
    valeur = valeur * volume
    gamma = valeur *100
    gamma = round (gamma, 3)
    return gamma

def vega(S,K,r,q,T,sigma,volume):
    r = r/100
    sigma = sigma/100
    q = q / 100
    valeur = S*np.sqrt(T)*np.exp(-q*(T))*norm.pdf((1/sigma*np.sqrt(T))*(np.log(S/K)+(r-q+0.5*sigma*sigma)*(T)),0,1)
    valeur = valeur * volume
    vega = valeur / 100
    vega = round (vega, 3)
    return vega

def thetaCall(S,K,r,q,T,sigma,volume):
    r = r/100
    sigma = sigma/100
    q = q / 100
    valeur = -((S*sigma*np.exp(-q*(T)))/(2*np.sqrt(T)))*norm.pdf((1/sigma*np.sqrt(T))*(np.log(S/K)+(r-q+0.5*sigma*sigma)*(T)),0,1)-r*np.exp(-r*(T))*K*norm.cdf(((1/sigma*np.sqrt(T))*(np.log(S/K)+(r-q+0.5*sigma*sigma)*(T)))-sigma*np.sqrt(T),0,1)+q*S*np.exp(-q*(T))*norm.cdf((1/sigma*np.sqrt(T))*(np.log(S/K)+(r-q+0.5*sigma*sigma)*(T)),0,1)
    valeur = valeur * volume
    theta = round (valeur, 3)
    return theta

def thetaPut(S,K,r,q,T,sigma,volume):
    r = r/100
    sigma = sigma/100
    q = q / 100
    valeur = -((S*sigma*np.exp(-q*(T)))/(2*np.sqrt(T)))*norm.pdf((1/sigma*np.sqrt(T))*(np.log(S/K)+(r-q+0.5*sigma*sigma)*(T)),0,1)+r*np.exp(-r*(T))*K*norm.cdf((-(1/sigma*np.sqrt(T))*(np.log(S/K)+(r-q+0.5*sigma*sigma)*(T)))+sigma*np.sqrt(T),0,1)-q*S*np.exp(-q*(T))*norm.cdf(-(1/sigma*np.sqrt(T))*(np.log(S/K)+(r-q+0.5*sigma*sigma)*(T)),0,1)	
    valeur = valeur * volume
    theta = round(valeur, 3)
    return theta

def rhoCall(S,K,r,q,T,sigma,volume):
    r = r/100
    sigma = sigma/100
    q = q / 100
    valeur = K*(T)*np.exp(-r*(T))*norm.cdf(((1/sigma*np.sqrt(T))*(np.log(S/K)+(r-q+0.5*sigma*sigma)*(T)))-sigma*np.sqrt(T),0,1)
    valeur = valeur * volume
    rho = round(valeur, 3)
    return rho

def rhoPut(S,K,r,q,T,sigma,volume):
    r = r/100
    sigma = sigma/100
    q = q / 100
    valeur = -K*(T)*np.exp(-r*(T))*norm.cdf((-(1/sigma*np.sqrt(T))*(np.log(S/K)+(r-q+0.5*sigma*sigma)*(T)))+sigma*np.sqrt(T),0,1)
    valeur = valeur * volume
    rho = round(valeur,3)
    return rho

#functions utilitaires
def indexByName (data, name):
    for i in range (len(data)):
        if name == data.loc[i, 'Name']:
            return i



#sortie de la page
app.layout = html.Div(
    #div de la page entiere
    html.Div([
        #div du titre
        html.Div([
            html.Div([
                html.H1(children ='Tool for risk gestion')
            ], className = 'seven columns'),

            html.Div([
                html.P('Name of the option'),
                dcc.Input(
                    placeholder = 'Enter name Option',
                    id = 'inputOption',
                    type = 'text',
                )
            ], className = 'two columns'),

            

            html.Div([
                html.P('Button'),
                html.Button(
                    'add',
                    id = 'Button',
                    n_clicks = 0
                )
            ], className = 'two columns')

        ], className = 'row'),

        #div de la ligne d'entrée de donnée (selecteur) soit un row
        html.Div([

            html.Div([
                html.P('Options'),
                dcc.Dropdown(
                    id = 'optionsName'
                )
            ], className = 'two columns'),

            html.Div([
                html.P('Choose'),
                dcc.RadioItems(
                    id = 'TypeO',
                    options=[
                        {'label': 'Call', 'value': 'Call'},
                        {'label': 'Put', 'value': 'Put'}],
                    value = 'Call',
                    labelStyle={'display': 'inline-block'}
                )  
            ], className = 'one columns'),
        ], className = 'one row'),

        html.Div([
            html.Div(id='output-container0', className = 'seven columns')
        ], className = 'one row'),

        html.Div([
            html.Div ([
                html.P('Volatility (%)'),
                dcc.Input(
                    id = 'volatility',
                    type = 'number'
                )
            ], className = 'two columns'),

            html.Div([
                html.P('Strike'),
                dcc.Input(
                    id = 'strike',
                    type = 'number'
                )
            ], className = 'two columns'),

            html.Div([
                html.P('Spot'),
                dcc.Input(
                    id = 'spot',
                    type = 'number'
                )
            ], className = 'two columns'),


            html.Div([
                html.P('Interest rate (%)'),
                dcc.Input(
                    id = 'interest rate',
                    type = 'number'
                )
            ], className = 'two columns'),


            html.Div([
                html.P('Maturity (days)'),
                dcc.Input(
                    id = 'maturity',
                    type = 'number'
                )
            ], className = 'two columns'),
            

            html.Div ([
                html.P('Volume'),
                dcc.Input(
                    id = 'volume',
                    type = 'number'
                )
            ], className = 'two columns')

        ], className = 'one row'),

        html.Div([
            html.Div(id='output-container1', className = 'six columns')
        ], className = 'one row'),
        

        html.Div([
            html.Div([
                html.P('Options\' Portefolio'),
                dash_table.DataTable(
                    id='tableDonnee',
                    columns=[{"name": i, "id": i} for i in donneeStockagePortefeuille.columns],
                    data=[]
                )
            ], className = 'twelve columns'),
        ], className = 'row'),

        html.Div([
            html.Div([
                html.P('Choose variations\' parameters'),
                dcc.Dropdown(
                    id='choixVariationsGreeks',
                    options=[
                        {'label':'Volatility', 'value':'Volatility'},
                        {'label':'Spot', 'value': 'Spot'},
                        {'label':'Strike', 'value': 'Strike'},
                        {'label':'Interest Rate', 'value':'Interest Rate'},
                        {'label':'Maturity', 'value':'Maturity'}],
                    value='Volatility'
                )
            ], className = 'three columns'),

            html.Div([
                html.P('Choose the variation (ex: 15 --> -15:+15)'),
                dcc.Input(
                    id='variationGreeks',
                    type='number',
                    value=0
                )
            ], className = 'four columns')
        ], className = 'row'),

        html.Div([
            html.Div([
                html.P('Total Greeks Portefolio'),
                dash_table.DataTable(
                    id='tableGreekSomme',
                    columns=[{"name": i, "id":i} for i in donneeTotalGreeks1.columns],
                    data=[]
                )
            ], className = 'twelve columns')
        ], className = 'row'),

        html.Div([
            html.Div([
                html.P('Choose a Greek'),
                dcc.Dropdown(
                    id='Greek',
                    options = [
                        {'label':'Delta', 'value':'Delta'},
                        {'label':'Gamma', 'value':'Gamma'},
                        {'label':'Theta', 'value':'Theta'},
                        {'label':'Rho', 'value':'Rho'},
                    ],
                    value='Delta'
                )
            ], className = 'two columns')
        ], className = 'row')
    ]))


#mise a jour de la page via callback


    
# Ajout dans la dataframe de stockage via un callback au bouton + ajout dans le dropdown
@app.callback(
    Output('optionsName', 'options'),
    [Input('Button', 'n_clicks')],
    [State('inputOption', 'value'),
    State('TypeO', 'value')]
)
def ajouterDataframe (n_clicks, value, typeOption): 
    if n_clicks > 0:    
        for i in range (len(donneeTest)):
            if value == donneeTest.loc[i,'Name']:
                dfa = pd.DataFrame([[0 for i in range (15)]], columns = ['Name', 'Type','Volume', 'Volatility', 'Strike', 'Spot', 'Interest rate', 'Maturity', 'Dividend', 'Price', 'Delta', 'Gamma', 'Theta', 'Rho', 'Vega'])
                dfa.loc[0,'Name']=value
                dfa.loc[0,'Type']='typeOption'
                dfa.loc[0,'Volume']=0
                dfa.loc[0,'Volatility']=donneeTest.loc[i,'Volatility']
                dfa.loc[0,'Strike']=donneeTest.loc[i,'Strike']
                dfa.loc[0,'Spot']=donneeTest.loc[i,'Spot']
                dfa.loc[0,'Interest rate']=donneeTest.loc[i,'Interest rate']
                dfa.loc[0,'Maturity']=0
                dfa.loc[0,'Dividend']=donneeTest.loc[i,'Dividend']
                dfa.loc[0,'Price']=0
                dfa.loc[0,'Delta']=0
                dfa.loc[0,'Gamma']=0
                dfa.loc[0,'Theta']=0
                dfa.loc[0,'Rho']=0
                dfa.loc[0,'Vega']=0
                global donneeStockagePortefeuille
                donneeCopy = donneeStockagePortefeuille.copy()
                donneeStockagePortefeuille = donneeCopy.append(dfa, ignore_index=True)
        options= [{'label': item, 'value': item} for item in (donneeStockagePortefeuille['Name'])]
        return options





#affichage vol
@app.callback(
    Output('volatility', 'value'),
    [Input('optionsName', 'value')]
)
def remplirVolatility(optionChoisi):
    for i in range (len(donneeStockagePortefeuille)):
        if optionChoisi == donneeStockagePortefeuille.loc[i,'Name']:
            return donneeStockagePortefeuille.loc[i,'Volatility'] 

#affichage strike
@app.callback(
    Output('strike', 'value'),
    [Input('optionsName', 'value')]
)
def remplirStrike(optionChoisi):
    for i in range (len(donneeStockagePortefeuille)):
        if optionChoisi == donneeStockagePortefeuille.loc[i,'Name']:
            return donneeStockagePortefeuille.loc[i,'Strike'] 

#affichage spot
@app.callback(
    Output('spot', 'value'),
    [Input('optionsName', 'value')]
)
def remplirSpot(optionChoisi):
   for i in range (len(donneeStockagePortefeuille)):
        if optionChoisi == donneeStockagePortefeuille.loc[i,'Name']:
            return donneeStockagePortefeuille.loc[i,'Spot']

#Affichage interest rate
@app.callback(
    Output('interest rate', 'value'),
    [Input('optionsName', 'value')]
)
def remplirIR(optionChoisi):
   for i in range (len(donneeStockagePortefeuille)):
        if optionChoisi == donneeStockagePortefeuille.loc[i,'Name']:
            return donneeStockagePortefeuille.loc[i,'Interest rate'] 





#Ce callback met a jour dans la dataframe de donnee le price et les greeks a chaque modification des cases
#J'ai laissé les print en bas. Cela permet de voir dans la console toute les modifications de la dataframe principale
#et des greeks. 

@app.callback(
    Output('tableDonnee', 'data'),
    [Input('volatility', 'value'),
    Input('strike', 'value'),
    Input('spot', 'value'),
    Input('interest rate', 'value'),
    Input('maturity', 'value'),
    Input('TypeO', 'value'),
    Input('volume', 'value')],
    [State('optionsName', 'value'),
    State('tableDonnee','data')]
)
def updateByVol(vol, strike, spot, ir, maturity, typeOption, volume, name, graphDonnee):
    if maturity is not None and vol is not None and strike is not None and spot is not None and ir is not None and name is not None and volume is not None:
        if maturity > 0: 
            #calcul de la maturité en jours
            maturityAnnee = maturity/365

            #recupération de l'index, type de l'option et des dividendes
            index = indexByName(donneeStockagePortefeuille, name)
            q = donneeStockagePortefeuille.loc[index, 'Dividend']

            #mise à jour du tableau avec les nouvelles valeurs entré

            donneeStockagePortefeuille.loc[index, 'Volatility'] = vol
            donneeStockagePortefeuille.loc[index, 'Strike'] = strike
            donneeStockagePortefeuille.loc[index, 'Spot'] = spot
            donneeStockagePortefeuille.loc[index, 'Interest rate'] = ir
            donneeStockagePortefeuille.loc[index, 'Maturity'] = maturity
            donneeStockagePortefeuille.loc[index, 'Type'] = typeOption
            donneeStockagePortefeuille.loc[index, 'Volume'] = volume



            #calcul des greeks et du prix de l'option
            donneeStockagePortefeuille.loc[index, 'Price'] = euro_vanilla_dividend(spot, strike, maturityAnnee, ir, q, vol, typeOption,volume)
            donneeStockagePortefeuille.loc[index, 'Gamma'] = gamma(spot, strike, ir, q, maturityAnnee, vol,volume)
            donneeStockagePortefeuille.loc[index, 'Vega'] = vega(spot, strike, ir, q, maturityAnnee, vol,volume)
            if typeOption == 'Call':
                donneeStockagePortefeuille.loc[index, 'Delta'] = deltaCall(spot, strike, ir, q, maturityAnnee, vol,volume)
                donneeStockagePortefeuille.loc[index, 'Theta'] = thetaCall(spot, strike, ir, q, maturityAnnee, vol,volume)
                donneeStockagePortefeuille.loc[index, 'Rho'] = rhoCall(spot, strike, ir, q, maturityAnnee, vol,volume)
            if typeOption == 'Put':
                donneeStockagePortefeuille.loc[index, 'Delta'] = deltaPut(spot, strike, ir, q, maturityAnnee, vol,volume)
                donneeStockagePortefeuille.loc[index, 'Theta'] = thetaPut(spot, strike, ir, q, maturityAnnee, vol,volume)
                donneeStockagePortefeuille.loc[index, 'Rho'] = rhoPut(spot, strike, ir, q, maturityAnnee, vol,volume)
        
        #retour des donnees pour l'affichage
        graphDonnee=donneeStockagePortefeuille.to_dict('rows')
    return graphDonnee


#Callbacks s'occupant de la partie "total des greeks"
@app.callback(
    Output('tableGreekSomme','data'),
    [Input('variationGreeks','value'),
    Input('tableDonnee', 'data'),
    Input('choixVariationsGreeks','value')],
    [State('tableGreekSomme', 'data')]
)
def callbackTotalGreeks(variation, dataTable, typeVariation, datatab):
    donneeTotalGreeks = pd.DataFrame(columns = ['Variation', 'Delta', 'Gamma', 'Theta', 'Rho', 'Vega'])
    donneePortefeuille = donneeStockagePortefeuille.copy()
    for i in range (variation*2+1):
        totalGreeks = pd.DataFrame([[0 for i in range (6)]], columns = ['Variation', 'Delta', 'Gamma', 'Theta', 'Rho', 'Vega'])
        totalGreeks.loc[0,'Variation'] = i-variation
        for j in range (donneePortefeuille.shape[0]):
            if typeVariation == 'Strike':
                typeO = donneePortefeuille.loc[j,'Type']
                donneePortefeuille.loc[j,'Strike'] = donneeStockagePortefeuille.loc[j,'Strike']+i-variation
                strike = donneePortefeuille.loc[j,'Strike']
                volume = donneePortefeuille.loc[j,'Volume']
                spot = donneePortefeuille.loc[j,'Spot']
                ir = donneePortefeuille.loc[j,'Interest rate']
                q = donneePortefeuille.loc[j,'Dividend']
                mat = donneePortefeuille.loc[j,'Maturity']
                matAnnee = mat/365
                vol = donneePortefeuille.loc[j,'Volatility']
                donneePortefeuille.loc[j,'Gamma'] = gamma(spot, strike, ir, q, matAnnee, vol,volume)
                donneePortefeuille.loc[j,'Vega'] = vega(spot, strike, ir, q, matAnnee, vol,volume)
                if typeO == 'Call':
                    donneePortefeuille.loc[j,'Delta'] = deltaCall(spot, strike, ir, q, matAnnee, vol,volume)
                    donneePortefeuille.loc[j,'Theta'] = thetaCall(spot, strike, ir, q, matAnnee, vol,volume)
                    donneePortefeuille.loc[j,'Rho'] = rhoCall(spot, strike, ir, q, matAnnee, vol,volume)
                if typeO == 'Put':
                    donneePortefeuille.loc[j,'Delta'] = deltaPut(spot, strike, ir, q, matAnnee, vol,volume)
                    donneePortefeuille.loc[j,'Theta'] = thetaPut(spot, strike, ir, q, matAnnee, vol,volume)
                    donneePortefeuille.loc[j,'Rho'] = rhoPut(spot, strike, ir, q, matAnnee, vol,volume)
                totalGreeks.loc[0,'Delta'] += donneePortefeuille.loc[j,'Delta']
                totalGreeks.loc[0,'Gamma'] += donneePortefeuille.loc[j,'Gamma']
                totalGreeks.loc[0,'Theta'] += donneePortefeuille.loc[j,'Theta']
                totalGreeks.loc[0,'Rho'] += donneePortefeuille.loc[j,'Rho']
                totalGreeks.loc[0,'Vega'] += donneePortefeuille.loc[j,'Vega']

            if typeVariation == 'Volatility':
                typeO = donneePortefeuille.loc[j,'Type']
                donneePortefeuille.loc[j,'Volatility'] = donneeStockagePortefeuille.loc[j,'Volatility']+(i-variation)/100
                strike = donneePortefeuille.loc[j,'Strike']
                spot = donneePortefeuille.loc[j,'Spot']
                volume = donneePortefeuille.loc[j,'Volume']
                ir = donneePortefeuille.loc[j,'Interest rate']
                q = donneePortefeuille.loc[j,'Dividend']
                mat = donneePortefeuille.loc[j,'Maturity']
                matAnnee = mat/365
                vol = donneePortefeuille.loc[j,'Volatility']
                donneePortefeuille.loc[j,'Gamma'] = gamma(spot, strike, ir, q, matAnnee, vol, volume)
                donneePortefeuille.loc[j,'Vega'] = vega(spot, strike, ir, q, matAnnee, vol, volume)
                if typeO == 'Call':
                    donneePortefeuille.loc[j,'Delta'] = deltaCall(spot, strike, ir, q, matAnnee, vol, volume)
                    donneePortefeuille.loc[j,'Theta'] = thetaCall(spot, strike, ir, q, matAnnee, vol, volume)
                    donneePortefeuille.loc[j,'Rho'] = rhoCall(spot, strike, ir, q, matAnnee, vol, volume)
                if typeO == 'Put':
                    donneePortefeuille.loc[j,'Delta'] = deltaPut(spot, strike, ir, q, matAnnee, vol, volume)
                    donneePortefeuille.loc[j,'Theta'] = thetaPut(spot, strike, ir, q, matAnnee, vol, volume)
                    donneePortefeuille.loc[j,'Rho'] = rhoPut(spot, strike, ir, q, matAnnee, vol, volume)
                totalGreeks.loc[0,'Delta'] += donneePortefeuille.loc[j,'Delta']
                totalGreeks.loc[0,'Gamma'] += donneePortefeuille.loc[j,'Gamma']
                totalGreeks.loc[0,'Theta'] += donneePortefeuille.loc[j,'Theta']
                totalGreeks.loc[0,'Rho'] += donneePortefeuille.loc[j,'Rho']
                totalGreeks.loc[0,'Vega'] += donneePortefeuille.loc[j,'Vega']

            if typeVariation == 'Spot':
                typeO = donneePortefeuille.loc[j,'Type']
                donneePortefeuille.loc[j,'Spot'] = donneeStockagePortefeuille.loc[j,'Spot']+i-variation
                strike = donneePortefeuille.loc[j,'Strike']
                spot = donneePortefeuille.loc[j,'Spot']
                ir = donneePortefeuille.loc[j,'Interest rate']
                q = donneePortefeuille.loc[j,'Dividend']
                volume = donneePortefeuille.loc[j,'Volume']
                mat = donneePortefeuille.loc[j,'Maturity']
                matAnnee = mat/365
                vol = donneePortefeuille.loc[j,'Volatility']
                donneePortefeuille.loc[j,'Gamma'] = gamma(spot, strike, ir, q, matAnnee, vol, volume)
                donneePortefeuille.loc[j,'Vega'] = vega(spot, strike, ir, q, matAnnee, vol, volume)
                if typeO == 'Call':
                    donneePortefeuille.loc[j,'Delta'] = deltaCall(spot, strike, ir, q, matAnnee, vol, volume)
                    donneePortefeuille.loc[j,'Theta'] = thetaCall(spot, strike, ir, q, matAnnee, vol, volume)
                    donneePortefeuille.loc[j,'Rho'] = rhoCall(spot, strike, ir, q, matAnnee, vol, volume)
                if typeO == 'Put':
                    donneePortefeuille.loc[j,'Delta'] = deltaPut(spot, strike, ir, q, matAnnee, vol, volume)
                    donneePortefeuille.loc[j,'Theta'] = thetaPut(spot, strike, ir, q, matAnnee, vol, volume)
                    donneePortefeuille.loc[j,'Rho'] = rhoPut(spot, strike, ir, q, matAnnee, vol, volume)
                totalGreeks.loc[0,'Delta'] += donneePortefeuille.loc[j,'Delta']
                totalGreeks.loc[0,'Gamma'] += donneePortefeuille.loc[j,'Gamma']
                totalGreeks.loc[0,'Theta'] += donneePortefeuille.loc[j,'Theta']
                totalGreeks.loc[0,'Rho'] += donneePortefeuille.loc[j,'Rho']
                totalGreeks.loc[0,'Vega'] += donneePortefeuille.loc[j,'Vega']
            
            if typeVariation == 'Interest Rate':
                typeO = donneePortefeuille.loc[j,'Type']
                donneePortefeuille.loc[j,'Interest rate'] = donneeStockagePortefeuille.loc[j,'Interest rate']+(i-variation)/100
                strike = donneePortefeuille.loc[j,'Strike']
                spot = donneePortefeuille.loc[j,'Spot']
                ir = donneePortefeuille.loc[j,'Interest rate']
                volume = donneePortefeuille.loc[j,'Volume']
                q = donneePortefeuille.loc[j,'Dividend']
                mat = donneePortefeuille.loc[j,'Maturity']
                matAnnee = mat/365
                vol = donneePortefeuille.loc[j,'Volatility']
                donneePortefeuille.loc[j,'Gamma'] = gamma(spot, strike, ir, q, matAnnee, vol, volume)
                donneePortefeuille.loc[j,'Vega'] = vega(spot, strike, ir, q, matAnnee, vol, volume)
                if typeO == 'Call':
                    donneePortefeuille.loc[j,'Delta'] = deltaCall(spot, strike, ir, q, matAnnee, vol, volume)
                    donneePortefeuille.loc[j,'Theta'] = thetaCall(spot, strike, ir, q, matAnnee, vol, volume)
                    donneePortefeuille.loc[j,'Rho'] = rhoCall(spot, strike, ir, q, matAnnee, vol, volume)
                if typeO == 'Put':
                    donneePortefeuille.loc[j,'Delta'] = deltaPut(spot, strike, ir, q, matAnnee, vol, volume)
                    donneePortefeuille.loc[j,'Theta'] = thetaPut(spot, strike, ir, q, matAnnee, vol, volume)
                    donneePortefeuille.loc[j,'Rho'] = rhoPut(spot, strike, ir, q, matAnnee, vol, volume)
                totalGreeks.loc[0,'Delta'] += donneePortefeuille.loc[j,'Delta']
                totalGreeks.loc[0,'Gamma'] += donneePortefeuille.loc[j,'Gamma']
                totalGreeks.loc[0,'Theta'] += donneePortefeuille.loc[j,'Theta']
                totalGreeks.loc[0,'Rho'] += donneePortefeuille.loc[j,'Rho']
                totalGreeks.loc[0,'Vega'] += donneePortefeuille.loc[j,'Vega']

            if typeVariation == 'Maturity':
                typeO = donneePortefeuille.loc[j,'Type']
                donneePortefeuille.loc[j,'Maturity'] = donneeStockagePortefeuille.loc[j,'Maturity']+i-variation
                strike = donneePortefeuille.loc[j,'Strike']
                spot = donneePortefeuille.loc[j,'Spot']
                ir = donneePortefeuille.loc[j,'Interest rate']
                volume = donneePortefeuille.loc[j,'Volume']
                q = donneePortefeuille.loc[j,'Dividend']
                mat = donneePortefeuille.loc[j,'Maturity']
                matAnnee = mat/365
                vol = donneePortefeuille.loc[j,'Volatility']
                donneePortefeuille.loc[j,'Gamma'] = gamma(spot, strike, ir, q, matAnnee, vol, volume)
                donneePortefeuille.loc[j,'Vega'] = vega(spot, strike, ir, q, matAnnee, vol, volume)
                if typeO == 'Call':
                    donneePortefeuille.loc[j,'Delta'] = deltaCall(spot, strike, ir, q, matAnnee, vol, volume)
                    donneePortefeuille.loc[j,'Theta'] = thetaCall(spot, strike, ir, q, matAnnee, vol, volume)
                    donneePortefeuille.loc[j,'Rho'] = rhoCall(spot, strike, ir, q, matAnnee, vol, volume)
                if typeO == 'Put':
                    donneePortefeuille.loc[j,'Delta'] = deltaPut(spot, strike, ir, q, matAnnee, vol, volume)
                    donneePortefeuille.loc[j,'Theta'] = thetaPut(spot, strike, ir, q, matAnnee, vol, volume)
                    donneePortefeuille.loc[j,'Rho'] = rhoPut(spot, strike, ir, q, matAnnee, vol, volume)
                totalGreeks.loc[0,'Delta'] += donneePortefeuille.loc[j,'Delta']
                totalGreeks.loc[0,'Gamma'] += donneePortefeuille.loc[j,'Gamma']
                totalGreeks.loc[0,'Theta'] += donneePortefeuille.loc[j,'Theta']
                totalGreeks.loc[0,'Rho'] += donneePortefeuille.loc[j,'Rho']
                totalGreeks.loc[0,'Vega'] += donneePortefeuille.loc[j,'Vega']

        mem = donneeTotalGreeks.copy()
        donneeTotalGreeks = mem.append(totalGreeks)
    datatab = donneeTotalGreeks.to_dict('rows')
    return datatab
        




if __name__ == '__main__':
    app.run_server(debug=True)


