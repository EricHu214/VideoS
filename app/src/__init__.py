from flask import Flask, request, redirect, jsonify, render_template, send_file
from app.src.scripts.video import stabilize
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
            smoothness = request.form.get('smoothness')

            stabilizedVideo = stabilize(video, smoothness)

            return send_file(
                stabilizedVideo,
                #attachment_filename="out.mp4",
                attachment_filename="stabilized_" + video.filename,
                as_attachment=True)


    return redirect('/')

@app.route('/test', methods=['GET'])
def test():
    return "hello world"
