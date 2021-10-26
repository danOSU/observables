import streamlit as st
import numpy as np
import pandas as pd
import time

st.title('Relativistic Heavy Ion Collision Simulation with \
Viscous Anisotropic Hydrodynamic')

st.write('Welcome! Use the sliders to the left of the page to find\
parameters that best fit the experimental data shown')

chart_data = pd.DataFrame(
     np.random.randn(20, 3),
     columns=['a', 'b', 'c'])

if st.checkbox('Show dataframe'):
    chart_data = pd.DataFrame(
       np.random.randn(20, 3),
       columns=['a', 'b', 'c'])

    st.write(chart_data)

df = pd.DataFrame({
  'first column': [1, 2, 3, 4],
  'second column': [10, 20, 30, 40]
})

option = st.sidebar.selectbox(
    'Which number do you like best?',
     df['first column'])

st.write(f'you selected {option}')

left_column, right_column = st.columns(2)
pressed = left_column.button('Press me?')
if pressed:
  right_column.write("Woohoo!")

expander = st.expander("FAQ")
expander.write("Here you could put in some really, really long explanations...")


slider_val_1 = st.sidebar.slider('shear viscosity',0.0,1.0,value=0.05, \
step = 0.5, help='select the viscosity value', format='%.1f')
st.write(f'value of the slide is {slider_val_1}')

with st.echo():
    st.write('this code is printed')

'Starting a long computation...'

# Add a placeholder
latest_iteration = st.empty()
bar = st.progress(0)

for i in range(100):
  # Update the progress bar with each iteration.
  latest_iteration.text(f'Iteration {i+1}')
  bar.progress(i + 1)
  time.sleep(0.1)

'...and now we\'re done!'
