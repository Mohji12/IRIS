import streamlit as st
import joblib
import pandas as pd

model = joblib.load("Logistic.joblib")

st.title('IRIS  Flower Classification')

# Taking Input From the User Interface

sepal_length = st.number_input('Sepal Length: ')

sepal_width = st.number_input('Sepal Width: ')

petal_length = st.number_input('Petal Length: ')

petal_width = st.number_input('Petal Width: ')


st.header("Your Input values")
df = pd.DataFrame({"sepal_length": [sepal_length],"sepal_width":[sepal_width],"petal_length": [petal_length],"petal_width":[petal_width]})
st.write(df)

prediction = model.predict([[sepal_length,sepal_width,petal_length,petal_width]])

if st.button("Predict"):
   st.subheader("Your Prediction")
   if prediction == 0:
      st.image("IRIS/setosa.png",width=300)
   elif prediction == 1:
      st.image("IRIS/Versicolor.png",width=300)
   else:
      st.image("IRIS/Vriginica.png",width=300)