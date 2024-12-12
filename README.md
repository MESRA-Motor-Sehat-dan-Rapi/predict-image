# Predict API

## Trigger the predict function from the model and get the API prediction endpoint :

This is a python code, so make sure you have python installed on your system.

1. Clone the repository then open it using your code editor.
2. Supposedly you have trained the model (from the [Machine-Learning](https://github.com/MESRA-Motor-Sehat-dan-Rapi) repository), download the model file with the **.h5** file format and name it as desired (to match with the scripts), then move it to the root directory of this project.
3. This code is using Google Cloud Storage, so you have to make your own GCS Bucket, make a folder named **predict_uploads** inside the bucket, get the credentials file (.json file) and name it "**mesra-credentials.json**" (to match with the scripts) then copy it to the root directory of this project.
4. Go to **main.py** edit the code in line 34, change '**mesra-bucket1**' with the name of the bucket that you created in the previous step.
5. Open terminal in the project root directory, then run `pip install -r requirements.txt` to install the dependencies.
6. Run the app using the command: `python main.py`.
7. By default, the server will run on the localhost with the port 5000, open [http://localhost:5000](http://localhost:5000) to view it in your browser.
8. If it shows 'OK' then you have successfully run the predict api.
9. The next step is to configure the backend service, you can find it in the [backend](https://github.com/MESRA-Motor-Sehat-dan-Rapi/back-end-mesra) repository.
