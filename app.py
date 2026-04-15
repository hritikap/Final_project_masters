import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


st.set_page_config(page_title="Healthacare Insurance Cost Prediction",layout='wide')
st.title("Healthcare Insurance Cost Analysis and Prediction")

df=pd.read_csv('data/insurance.csv')

tab1,tab2= st.tabs(['Exploratory Data Analysis','Cost Predictor'])

with tab1:
    col1,col2=st.columns(2)
    with col1:
        fig,ax=plt.subplots()
        sns.boxplot(x="smoker",y="charges",data=df,palette='pastel')
        ax.set_title('Charges by Smoking Status')
        st.pyplot(fig)

        with col2:
            fif,ax=plt.subplots()
            sns.scatterplot(x='age',y='charges',hue='smoker',data=df,alpha=0.7,ax=ax)
            ax.set_title('Age vs Charges (Colored by Smoker)')
            st.pyplot(fig)

with tab2:
    st.subheader('Predict your Insurance Cost')
    age=st.slider("Age",18,65,30)
    bmi=st.slider("BMI",15.0,55.0,25.0)
    children=st.selectbox("Number of Children",[0,1,2,3,4,5])
    smoker=st.selectbox("Smoker",["Yes","No"])
    sex=st.selectbox("Sex",["male","female"])
    region=st.selectbox("Region",["southwest","southeast","northwest","northeast"])

    if st.button("Predict"):
        st.info("Load your saved model here and make a prediction !")
