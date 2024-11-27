# Importing essential libraries and modules

from flask import Flask, render_template, request, redirect, jsonify
from markupsafe import Markup
import numpy as np
import pandas as pd
from utils.disease import disease_dic
from utils.fertilizer import fertilizer_dic
import io
import torch
from torchvision import transforms
from PIL import Image
from utils.model import ResNet9
# ==============================================================================================

# -------------------------LOADING THE TRAINED MODELS -----------------------------------------------

# Loading plant disease classification model

disease_classes = ['Apple___Apple_scab',
                   'Apple___Black_rot',
                   'Apple___Cedar_apple_rust',
                   'Apple___healthy',
                   'Blueberry___healthy',
                   'Cherry_(including_sour)___Powdery_mildew',
                   'Cherry_(including_sour)___healthy',
                   'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                   'Corn_(maize)___Common_rust_',
                   'Corn_(maize)___Northern_Leaf_Blight',
                   'Corn_(maize)___healthy',
                   'Grape___Black_rot',
                   'Grape___Esca_(Black_Measles)',
                   'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                   'Grape___healthy',
                   'Orange___Haunglongbing_(Citrus_greening)',
                   'Peach___Bacterial_spot',
                   'Peach___healthy',
                   'Pepper,_bell___Bacterial_spot',
                   'Pepper,_bell___healthy',
                   'Potato___Early_blight',
                   'Potato___Late_blight',
                   'Potato___healthy',
                   'Raspberry___healthy',
                   'Soybean___healthy',
                   'Squash___Powdery_mildew',
                   'Strawberry___Leaf_scorch',
                   'Strawberry___healthy',
                   'Tomato___Bacterial_spot',
                   'Tomato___Early_blight',
                   'Tomato___Late_blight',
                   'Tomato___Leaf_Mold',
                   'Tomato___Septoria_leaf_spot',
                   'Tomato___Spider_mites Two-spotted_spider_mite',
                   'Tomato___Target_Spot',
                   'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                   'Tomato___Tomato_mosaic_virus',
                   'Tomato___healthy']

disease_model_path = 'models/plant_disease_model.pth'
disease_model = ResNet9(3, len(disease_classes))
disease_model.load_state_dict(torch.load(
    disease_model_path, map_location=torch.device('cpu')))
disease_model.eval()


# Loading crop recommendation model
# =========================================================================================

# Custom functions for calculations
# def weather_fetch(city_name):
#     """
#     Fetch and returns the temperature and humidity of a city
#     :params: city_name
#     :return: temperature, humidity
#     """
#     api_key = config.weather_api_key
#     base_url = "http://api.openweathermap.org/data/2.5/weather?"

#     complete_url = base_url + "appid=" + api_key + "&q=" + city_name
#     response = requests.get(complete_url)
#     x = response.json()

#     if x["cod"] != "404":
#         y = x["main"]

#         temperature = round((y["temp"] - 273.15), 2)
#         humidity = y["humidity"]
#         return temperature, humidity
#     else:
#         return None

def predict_image(img, model=disease_model):
    """
    Transforms image to tensor and predicts disease label
    :params: image
    :return: prediction (string)
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])
    image = Image.open(io.BytesIO(img))
    img_t = transform(image)
    img_u = torch.unsqueeze(img_t, 0)

    # Get predictions from model
    yb = model(img_u)
    # Pick index with highest probability
    _, preds = torch.max(yb, dim=1)
    prediction = disease_classes[preds[0].item()]
    # Retrieve the class label
    return prediction

# ===============================================================================================
# ------------------------------------ FLASK APP -------------------------------------------------

app = Flask(__name__)
# render home page

# # API URL and Key (Replace with your actual URL if needed)
# API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key=AIzaSyBU7F_0jauYMwQ1KrcHcWyow3vGUlcSj_k"  # Change this to your actual Gemini API URL
# API_KEY = 'AIzaSyBU7F_0jauYMwQ1KrcHcWyow3vGUlcSj_k'  # Use your actual API Key

# Home route
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    # Get the incoming data from the frontend
    data = request.get_json()
    user_message = data.get('message')

    # Check if the message is not empty
    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    # Generate the response based on the user's message
    bot_response = generate_bot_response(user_message)

    return jsonify({"response": bot_response})

def generate_bot_response(user_message):
    # Example responses
    if "health" in user_message.lower():
        return "Farmers often face health risks from pesticides, heat stress, and musculoskeletal issues. Stay safe!"
    elif "pesticide" in user_message.lower():
        return "Pesticide exposure can lead to headaches, nausea, and skin problems. Always wear protective gear!"
    elif "protect" in user_message.lower():
        return " Farmers should wear protective clothing like gloves, masks, and goggles when applying pesticides. It's also important to work in well-ventilated areas and follow safety instructions on pesticide labels."
    elif "problem" in user_message.lower():
        return " Farmers often experience back pain, joint problems, and repetitive strain injuries due to heavy lifting, bending, and prolonged periods of standing or sitting. Regular exercise, proper lifting techniques, and ergonomic equipment can help prevent these issues."
    elif "heat" in user_message.lower():
        return "Symptoms include dizziness, nausea, rapid heartbeat, confusion, and excessive sweating. If a farmer experiences these symptoms, they should seek medical attention immediately and cool down in a shaded area."
    elif "yellow" in user_message.lower():
        return "Yellowing leaves may indicate a nutrient deficiency, particularly nitrogen or iron. It can also be a sign of overwatering, poor drainage, or root damage.Solution:Nitrogen deficiency: Apply a nitrogen-rich fertilizer like urea or compost.Iron deficiency: Consider using iron chelate or other iron supplements.Ensure proper drainage and avoid overwatering."
    elif "weeds" in user_message.lower():
        return " Weeds compete with crops for nutrients, light, and water, and can significantly reduce yields.Solution:Manual removal: Hand-pull weeds, especially when they’re young.Mulching: Apply mulch to suppress weed growth and retain moisture.Herbicides: Use targeted herbicides if the weed problem is severe. Always follow safety guidelines for chemical use.Cover crops: Plant cover crops like clover or rye to crowd out weeds"
    else:
        return "I'm here to help with farming health-related questions. Please ask another question."


@ app.route('/chat')
def chatbot():
    title = 'Harvestify - Crop Recommendation'
    return render_template('chat.html', title=title)

# render fertilizer recommendation form page


@ app.route('/fertilizer')
def fertilizer_recommendation():
    title = 'Harvestify - Fertilizer Suggestion'

    return render_template('fertilizer.html', title=title)

# ===============================================================================================

@ app.route('/fertilizer-predict', methods=['POST'])
def fert_recommend():
    title = 'Harvestify - Fertilizer Suggestion'

    crop_name = str(request.form['cropname'])
    N = int(request.form['nitrogen'])
    P = int(request.form['phosphorous'])
    K = int(request.form['pottasium'])
    # ph = float(request.form['ph'])

    df = pd.read_csv(r'C:\Users\Sarthak\Desktop\Harvestify\app\Data\fertilizer.csv')

    nr = df[df['Crop'] == crop_name]['N'].iloc[0]
    pr = df[df['Crop'] == crop_name]['P'].iloc[0]
    kr = df[df['Crop'] == crop_name]['K'].iloc[0]

    n = nr - N
    p = pr - P
    k = kr - K
    temp = {abs(n): "N", abs(p): "P", abs(k): "K"}
    max_value = temp[max(temp.keys())]
    if max_value == "N":
        if n < 0:
            key = 'NHigh'
        else:
            key = "Nlow"
    elif max_value == "P":
        if p < 0:
            key = 'PHigh'
        else:
            key = "Plow"
    else:
        if k < 0:
            key = 'KHigh'
        else:
            key = "Klow"

    response = Markup(str(fertilizer_dic[key]))

    return render_template('fertilizer-result.html', recommendation=response, title=title)

# render disease prediction result page

@app.route('/disease-predict', methods=['GET', 'POST'])
def disease_prediction():
    title = 'Harvestify - Disease Detection'

    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return render_template('disease.html', title=title)
        try:
            img = file.read()

            prediction = predict_image(img)

            prediction = Markup(str(disease_dic[prediction]))
            return render_template('disease-result.html', prediction=prediction, title=title)
        except:
            pass
    return render_template('disease.html', title=title)

# ===============================================================================================
if __name__ == '__main__':
    app.run(debug=True)