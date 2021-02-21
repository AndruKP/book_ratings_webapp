import flask
import pickle
import sklearn
import numpy as np
import pandas as pd
from urllib import request as urllib2
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
import tensorflow as tf
WIDTH = 300
HEIGHT = 450

f_model = open(f'models/tree.pkl', 'rb')  # TODO: make model
f_nn = open(f'models/saved_model.pb','rb')

model = pickle.load(f_model)
nn = tf.keras.models.load_model

app = flask.Flask(__name__, template_folder='templates')


@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return flask.render_template('main.html')

    if flask.request.method == 'POST':
        book_title = flask.request.form['book_title']
        book_image_url = flask.request.form['book_image_url']
        book_pages = np.log10(int(flask.request.form['book_pages']) + 1)
        book_review_count = np.log10(int(flask.request.form['book_review_count']) + 1)
        book_rating_count = np.log10(int(flask.request.form['book_rating_count']) + 1)

        im = Image.open(BytesIO(urllib2.urlopen(book_image_url).read()))
        im = im.resize((WIDTH, HEIGHT), Image.ANTIALIAS)
        im.save('models/image.png')
        im = plt.imread('models/image.png', format='png').reshape(1,450,300,3)

        nn_prediction = nn.predict(im)

        input_variables = pd.DataFrame([[book_pages, book_review_count, book_rating_count]],
                                           columns=['book_pages', 'book_review_count', 'book_rating_count'],
                                           dtype=float)
        prediction = model.predict(input_variables)[0]
        prediction = np.round((prediction+nn_prediction)/2, 2)
        return flask.render_template('main.html',
                                         original_input={
                                             #'book_pages': book_pages,
                                             #'book_review_count': book_review_count,
                                             #'book_rating_count': book_rating_count
                                             },
                                         result=prediction)


if __name__ == '__main__':
    app.run()
