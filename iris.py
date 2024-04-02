import streamlit as st
import pickle
import pandas as pd

model = pickle.load(open('Decision_Tree.pkl','rb'))

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
    st.write(prediction)