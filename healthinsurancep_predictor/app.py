from flask import Flask, app, render_template, url_for, request, redirect
import numpy as np
import pickle
import logging

app = Flask(__name__)
logging.basicConfig(filename="log\\log_files\\test.log", format='%(asctime)s %(message)s')
logger=logging.getLogger()
logger.setLevel(logging.DEBUG)

model = pickle.load(open('best-model_gbr.pkl', 'rb'))





@app.route('/')
def Home():
    return render_template('index.html')


@app.route("/predict", methods=['POST', 'GET'])
def predict():
  features = [int(x) for x in request.form.values()]

  print(features)
  final = np.array(features).reshape((1, 11))
  print(final)
  logger.log(1,"user input")
  predicted_value = model.predict(final)[0]
  print(predicted_value)
  logger.log(2,'predicted_value')
  if predicted_value < 0:
    return render_template('index.html', predicted_value='Something is wrong check logs')
  else:
    return render_template('result.html', predicted_value=' {0:.3f}'.format(predicted_value))


if __name__ == "__main__":
  app.run(debug=True)
