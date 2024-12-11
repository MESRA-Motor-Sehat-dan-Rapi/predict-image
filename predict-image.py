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


def req(y_true, y_pred):
    req = tf.metrics.req(y_true, y_pred)[1]
    tf.keras.backend.get_session().run(tf.local_variables_initializer())
    return req

model = load_model('sparepart.h5', custom_objects={'req': req})


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            image_bucket = storage_client.get_bucket(
                'mesra-bucket')
            filename = request.json['filename']
            img_blob = image_bucket.blob('predict_uploads/' + filename)
            img_path = BytesIO(img_blob.download_as_bytes())
        except Exception:
            respond = jsonify({'message': 'Error loading image file'})
            respond.status_code = 400
            return respond

        img = tf.keras.utils.load_img(img_path, target_size=(224, 224))
        x = tf.keras.utils.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        images = np.vstack([x])

        # model predict
        pred_motor = model.predict(images)
        # find the max prediction of the image
        maxx = pred_motor.max()

        spareparts  = [ 'spion', 'knalpot', 'spion_rusak', 'motor_lecet', 'honda_beat_biru_putih', 'honda_beat_hijau', 'honda_beat_hitam', 'honda_beat_silver','honda_vario_hitam', 'honda_vario_putih','yamaha_aerox_hitam', 'yamaha_aerox_kuning', 'yamaha_aerox_putih','yamaha_nmax_hitam', 'yamaha_nmax_merah', 'yamaha_nmax_putih', 'plat_nomor']

        # for respond output from prediction if predict <=0.4
        if maxx <= 0.75:
            respond = jsonify({
                'message': 'Spareparts tidak terdeteksi'
            })
            respond.status_code = 400
            return respond

        result = {
            "Spareparts": spareparts[np.argmax(pred_motor)],
        }

        respond = jsonify(result)
        respond.status_code = 200
        return respond

    return 'OK'


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')