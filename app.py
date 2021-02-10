import numpy as np
from flask import Flask, request, jsonify, render_template
import json
import requests
from config import *
from functions import extract_keywords

app = Flask(__name__)

TOPIC = 'Data'
NEWS_API_URL = "http://newsapi.org/v2/everything?q="+TOPIC+"&from=2021-01-01&to=2021-01-29&sortBy=popularity&language=en&apiKey=" + NEWS_API_KEY
URL = "http://newsapi.org/v2/everything?domains=datasciencecentral&from=2021-01-01&to=2021-01-29&sortBy=popularity&language=en&apiKey=" + NEWS_API_KEY
NEWS_API_URL_TOP = "https://newsapi.org/v2/top-headlines?q=Data&Science&sortBy=publishedAt&pageSize=100&language=en&apiKey=" + NEWS_API_KEY

@app.route("/")
def hello():
    return "Hello World!"

@app.route('/dashboard/')
def dashboard():
    return render_template("dashboard.html")

@app.route('/api/news/')
def news():
    response = requests.get(NEWS_API_URL_TOP)
    content = json.loads(response.content.decode('utf-8'))

    if response.status_code != 200:
        return jsonify({
            'status': 'error',
            'message': 'La requête à l\'API météo n\'a pas fonctionné. Voici le message renvoyé par l\'API : {}'.format(content['message'])
        }), 500

    keywords, articles = extract_keywords(content['articles'])
        
    return jsonify({
      'status': 'ok', 
      'data': {
          'keywords': keywords[:200],
          'articles': articles
      }
    })


if __name__ =="__main__": 
    app.run(debug=True)
