# -*- coding: utf-8 -*-
import os
from image_classificator.functions import result, test_image_in_best_model
from flask import Flask, render_template, session, redirect, url_for, request
from flask_uploads import UploadSet, configure_uploads, IMAGES, patch_request_class
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import SubmitField, SelectField
from wtforms.validators import DataRequired

app = Flask(__name__)
app.config['SECRET_KEY'] = 'I have a dream'
app.config['UPLOADED_PHOTOS_DEST'] = os.getcwd()

photos = UploadSet('photos', IMAGES)
configure_uploads(app, photos)
patch_request_class(app)  # set maximum file size, default is 16MB



class UploadForm(FlaskForm):
    photo = FileField(validators=[FileAllowed(photos, u'Image only!'), FileRequired(u'File was empty!')])
    feature_choice = SelectField(u'Escolha seu descritor:',
                          choices=[('histogram', 'Histograma'), ('glcm', 'GLCM'),
                                   ('hog', 'HOG'), ('haralick', 'Haralick') ])
    submit = SubmitField(u'Upload')



@app.route('/', methods=['GET', 'POST'])
def upload_file():

    form = UploadForm()

    if form.validate_on_submit():
        session['feature_choice'] = form.feature_choice.data
        filename = photos.save(form.photo.data)
        session['file_name'] = filename
        session['file_url'] = photos.url(filename)

        return redirect(url_for('result_upload', file_name=session['file_name'], feature_choice=session['feature_choice']))
    else:
        session['file_url'] = None

    return render_template('upload.html', form=form)

@app.route('/result_upload')
def result_upload():
    file_name = request.args['file_name']  # counterpart for url_for()
    file_name = session['file_name']
    feature_choice = request.args['feature_choice']  
    feature_choice = session['feature_choice']
    resultado = test_image_in_best_model(file_name, feature_choice)
    return render_template('result_upload.html', file_url=resultado)

if __name__ == '__main__':
    app.run()