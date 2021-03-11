# -*- coding: utf-8 -*-
from flask import Flask, render_template, request
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
from tensorflow.keras import optimizers
from keras.models import load_model
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# from keras.backend import tensorflow_backend as K
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# K.set_session(tf.Session(config=config))

global avg_temp
global min_temp
global max_temp
global rain_fall
global model
global out

app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def index():
	if request.method == 'GET':
		return render_template('index.html')
	if request.method == 'POST':
		avg_temp = float(request.form['avg_temp'])
		min_temp = float(request.form['min_temp'])
		max_temp = float(request.form['max_temp'])
		rain_fall = float(request.form['rain_fall'])

		if str(request.form['mode']) == 'vegi': #배추값 예측 
			model = load_model('C:\TensorWeather\\tsWeather_last.h5')

			data = ((avg_temp,min_temp,max_temp,rain_fall),) #튜플을 사용해서 온도값을 저장

			arr = np.array(data,dtype=np.float32) #numpy를 사용하여서 원하는 데이터의 형태로 변환

			x_data = arr[0:4] #슬라이싱 문법으로 원하는 데이터를 추출 

			out = str(model.predict(x_data))
			return render_template('index.html', price = out)

		if str(request.form['mode']) == 'dust': #미세먼지 발생율 예측 

			model = load_model('C:\TensorWeather\\TW_dust.h5')

			data = ((rain_fall),)

			arr = np.array(data,dtype=np.float32) #numpy를 사용하여서 원하는 데이터의 형태로 변환

			x_data = arr[0:1] #슬라이싱 문법으로 원하는 데이터를 추출 

			dus = str(model.predict(x_data)*100)
			return render_template('index.html', dust = dus)

		if str(request.form['mode']) == 'acc': #교통사고 발생율 예측 
			model = load_model('C:\TensorWeather\\TW_accident_acc.h5') #교통사고 예측 모델 담기

			data = ((avg_temp,min_temp,max_temp,rain_fall),) #튜플을 사용해서 온도값을 저장

			arr = np.array(data,dtype=np.float32) #numpy를 사용하여서 원하는 데이터의 형태로 변환

			x_data = arr[0:4] #슬라이싱 문법으로 원하는 데이터를 추출 

			out_acc = str(model.predict(x_data)) # 모델 예측 

			model = load_model('C:\TensorWeather\\TW_accident_dead.h5') #교통사고 사망 예측 모델담기

			out_dead = str(model.predict(x_data)) #모델 예측 

			return render_template('index.html',acc = out_acc, dead = out_dead)


if __name__ == '__main__':
   app.run(debug = True) 