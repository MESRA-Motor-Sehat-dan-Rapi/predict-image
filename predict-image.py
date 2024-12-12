import os
from google.cloud import storage
import tensorflow as tf
from io import BytesIO
from flask import Flask, request, jsonify
from keras.models import load_model
import numpy as np
from tensorflow.keras.applications.mobilenet import preprocess_input

app = Flask(__name__)
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'mesra-credentials.json'
storage_client = storage.Client()

# Custom metric function
def req(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

# Load the model
model = load_model('sparepart.h5', custom_objects={'req': req})

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Load image from GCS
            image_bucket = storage_client.bucket('mesra-bucket')
            filename = request.json['filename']
            img_blob = image_bucket.blob(f'predict_uploads/{filename}')
            img_path = BytesIO(img_blob.download_as_bytes())
        except Exception as e:
            respond = jsonify({'message': f'Error loading image file: {str(e)}'})
            respond.status_code = 400
            return respond

        try:
            # Preprocess image
            img = tf.keras.utils.load_img(img_path, target_size=(224, 224))
            x = tf.keras.utils.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            images = np.vstack([x])

            # Predict using the model
            pred_motor = model.predict(images)

            # Validate prediction shape
            spareparts = [
                'spion', 'knalpot', 'spion_rusak', 'motor_lecet',
                'honda_beat_biru_putih', 'honda_beat_hijau', 'honda_beat_hitam',
                'honda_beat_silver', 'honda_vario_hitam', 'honda_vario_putih',
                'yamaha_aerox_hitam', 'yamaha_aerox_kuning', 'yamaha_aerox_putih',
                'yamaha_nmax_hitam', 'yamaha_nmax_merah', 'yamaha_nmax_putih',
                'plat_nomor'
            ]

            if len(pred_motor.shape) != 2 or pred_motor.shape[1] != len(spareparts):
                respond = jsonify({'message': 'Prediction output shape mismatch'})
                respond.status_code = 500
                return respond

            maxx = pred_motor.max()
            if maxx <= 0.75:
                respond = jsonify({'message': 'Spareparts tidak terdeteksi'})
                respond.status_code = 400
                return respond

            result = {"Spareparts": spareparts[np.argmax(pred_motor)]}
            respond = jsonify(result)
            respond.status_code = 200
            return respond
        except Exception as e:
            respond = jsonify({'message': f'Error during prediction: {str(e)}'})
            respond.status_code = 500
            return respond

    return 'OK'

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
