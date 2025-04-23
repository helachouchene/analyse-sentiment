from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Définir le dossier d'uploads
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Charger le modèle depuis le dossier parent (../models)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'vgg16_finetuned.keras')
model = load_model(MODEL_PATH)

# Liste des émotions
class_names = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    filename = None

    if request.method == "POST":
        file = request.files["image"]

        if file and file.filename != "":
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Charger et préparer l'image
            img = image.load_img(filepath, target_size=(48, 48))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0

            # Prédire
            preds = model.predict(img_array)
            prediction = class_names[np.argmax(preds)]

    return render_template("index.html", prediction=prediction, image=filename)

if __name__ == "__main__":
    app.run(debug=True)
