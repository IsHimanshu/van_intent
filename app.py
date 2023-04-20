from flask import Flask, request, render_template
import tensorflow as tf

myArray = ['AddToPlaylist', 'BookRestaurant', 'GetWeather', 'PlayMusic', 'RateBook', 'SearchCreativeWork', 'SearchScreeningEvent']




app = Flask(__name__)

# Load your deep learning model here
# ...

model = tf.keras.models.load_model('intent_(electra)bert.h5')

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get_intent", methods=["POST"])
def get_intent():
    # Get the text input from the form data
    text_input = request.form["text-input"]
    result = model.predict([str(text_input)])

    # Return the predicted intent
    return "Intent: " + result

if __name__ == "__main__":
    app.run(debug=True)

