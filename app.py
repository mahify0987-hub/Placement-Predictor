import gradio as gr
import pickle

clf = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

def predict(iq, cgpa):
    input_data = [[cgpa, iq]]
    input_data = scaler.transform(input_data)

    result = clf.predict(input_data)

    if result[0] == 1:
        return "Placement Ho Jaayega 🎉"
    else:
        return "Placement Nahi Hoga 😔"

demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="Enter IQ of the student"),
        gr.Number(label="Enter CGPA of the student")
    ],
    outputs=gr.Textbox(label="Prediction"),
    title="Placement Predictor"
)

demo.launch()
