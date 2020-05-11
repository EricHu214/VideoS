from flask import Flask, request, redirect, jsonify, render_template, send_file
from src.scripts.video import stabilize
import os


# init
app = Flask(__name__)

#app.config["VIDEO_UPLOADS"] = "C:/Users/Eric/Desktop/side projects/VideoS/app/src/static/uploads/"
#app.config["VIDEO_UPLOADS"] = "/mnt/c/Users/Eric/Desktop/side-projects/VideoS-unix/VideoS/app/src/static/uploads/"

@app.route('/', methods=['GET'])
def home():
    return render_template('upload.html')

@app.route('/upload-video', methods=['POST'])
def uploadVideo():
    redirect('/')
    if request.method == "POST":
        if request.files:
            video = request.files['video']
            stabilizedVideo = stabilize(video)

            return send_file(
                stabilizedVideo,
                attachment_filename="out.mp4",#attachment_filename="stabilized_" + video.filename,
                as_attachment=True)


    return redirect('/')

@app.route('/test', methods=['GET'])
def test():
    return "hello world"
