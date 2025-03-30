# źródło danych [https://www.kaggle.com/c/titanic/](https://www.kaggle.com/c/titanic)

import streamlit as st
import pickle
import sys
from datetime import datetime
startTime = datetime.now()
# import znanych nam bibliotek

import pathlib
from pathlib import Path

temp = pathlib.PosixPath if (sys.platform == "darwin") else pathlib.WindowsPath
#pathlib.PosixPath = pathlib.WindowsPath

filename = "model.sv"
model = pickle.load(open(filename,'rb'))
# otwieramy wcześniej wytrenowany model

sex_d = {0: "Kobieta", 1: "Męzczyzna"}
chestpaintype_d = {0: "ASY", 1: "ATA", 2: "MAP"}
restingecg_d = {0: "LVH", 1: "Normal", 2: "ST"}
exercise_angina_d = {0 : "No", 1: "Yes"}
st_slope_d = {0 : "Down", 1: "Flat", 2: "Up"}
fastingbs_d = {0 : "No", 1: "Yes"}

# pclass_d = {1:"Pierwsza",2:"Druga", 3:"Trzecia"}
# embarked_d = {0:"Cherbourg", 1:"Queenstown", 2:"Southampton"}
# sex_d = {0: "Męzczyzna", 1: "Kobieta"}
# o ile wcześniej kodowaliśmy nasze zmienne, to teraz wprowadzamy etykiety z ich nazewnictwem

def main():

	page_title = "Predykcja choroby serca"
	panel_title = "Kryteria"
	heart_image = "https://www.rodzinazdrowia.pl/Data/Thumbs/_public/zdrowie/uklad-krwionosny/kiedy-nasze-serce-choruje---chor/MTE3MHg2OTg,choroba-wiencowa-objawy.jpg"

	st.set_page_config(page_title=page_title)
	overview = st.container()
	left, right = st.columns(2)
	prediction = st.container()

	st.image(heart_image)

	with overview:
		st.title(panel_title)

	with left:
		sex_radio = st.radio( "Sex", list(sex_d.keys()), index = 1, format_func=lambda x : sex_d[x] )
		chestpaintype_radio = st.radio("Chest Pain Type", list(chestpaintype_d.keys()), index= 0, format_func=lambda x : chestpaintype_d[x])
		restingecg_radio = st.radio("Resting ECG", list(restingecg_d.keys()), index= 0, format_func=lambda x : restingecg_d[x])
		exercise_angina_radio = st.radio("Exercise Angina", list(exercise_angina_d.keys()), index= 0, format_func=lambda x : exercise_angina_d[x])
		st_slope_radio = st.radio("ST Slope", list(st_slope_d.keys()), index= 0, format_func=lambda x : st_slope_d[x])
		fastingbs_radio = st.radio("Fasting BS", list(fastingbs_d.keys()), index= 0, format_func=lambda x : fastingbs_d[x])

	with right:
		age_slider = st.slider("Age", value=1, min_value=1, max_value=100)
		restingbp_slider = st.slider("Resting BP", min_value=0, max_value=200)
		cholesterol_slider = st.slider("Cholesterol", min_value=0, max_value=610)
		maxhr_slider = st.slider("Max HR", min_value=60, max_value=210, step=1)	
		oldpeak_slider = st.slider("Oldpeak", min_value=-3.0, max_value=7.0, step=0.1)

	data = [[age_slider, sex_radio,  chestpaintype_radio, restingbp_slider, cholesterol_slider, fastingbs_radio, restingecg_radio, maxhr_slider,exercise_angina_radio,oldpeak_slider,st_slope_radio]]
	print(f"Dane przekazane do modelu: {data}")
	heart_rate = model.predict(data)
	s_confidence = model.predict_proba(data)

	with prediction:
		st.subheader("Czy taka osoba będzie miała zawał?")
		st.subheader(("Tak" if heart_rate[0] == 1 else "Nie"))
		st.write("Pewność predykcji {0:.2f} %".format(s_confidence[0][heart_rate][0] * 100))

if __name__ == "__main__":
    main()
