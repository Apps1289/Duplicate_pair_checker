import streamlit as st
import helper
import pickle

model1 = pickle.load(open('model.pkl','rb'))
model = pickle.load(open('w2v.pkl','rb'))

st.header('Check Duplicate Question Pairs')

q1 = st.text_input('Enter question 1')
q2 = st.text_input('Enter question 2')

if st.button('Find'):
    query = helper.query_point(q1,q2,model)
    result = model1.predict(query)[0]

    if result:
        st.header('Duplicate')
    else:
        st.header('Not Duplicate')