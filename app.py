import os
import tensorflow as tf
import gradio as gr
import numpy as np


loaded_model = tf.keras.saving.load_model('./model/linear_regressor.keras')

def test_ml_model(input):
    result = loaded_model.predict(([input]))
    return (f'predicted: {result}')

demo = gr.Interface(fn=test_ml_model, inputs=gr.Slider(0, 100, step=1), 
                    outputs="text",
                    description="A sample linear regressor solution.",
                    title='Synthetic Data Linear Regressor Solution')
    
demo.launch() 