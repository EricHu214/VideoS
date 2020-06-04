from flask import Flask, request, redirect, jsonify, render_template, send_file
from app.src.modules import stabilize, stabilizeYoutube
import os


# init
app = Flask(__name__)


@app.route('/', methods=['GET'])
def home():
    return render_template('upload.html')

@app.route('/upload-video', methods=['POST'])
def uploadVideo():
    redirect('/')
    if request.method == "POST":
        if request.files and request.files['video']:
            video = request.files['video']

            if video.content_type == "video/mp4":
                cropPercentage = request.form.get('crop_percentage')
                stabilizedVideo = stabilize(video, cropPercentage)

                return send_file(
                    stabilizedVideo,
                    #attachment_filename="out.mp4",
                    attachment_filename="stabilized_" + video.filename,
                    as_attachment=True,
                    mimetype='video/mp4')

        elif request.form.get('video-url'):
            cropPercentage = request.form.get('crop_percentage')
            stabilizedVideo = stabilizeYoutube(request.form.get('video-url'), cropPercentage)

            return send_file(
                stabilizedVideo,
                attachment_filename="stabilized.mp4",
                #attachment_filename="stabilized_" + video.filename,
                as_attachment=True,
                mimetype='video/mp4')


    return redirect('/')

@app.route('/test', methods=['GET'])
def test():
    return "hello world"
