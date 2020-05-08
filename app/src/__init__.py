from flask import Flask, request, redirect, jsonify, render_template, send_file
import os


# init
app = Flask(__name__)
#app.config["VIDEO_UPLOADS"] = "C:/Users/Eric/Desktop/side projects/VideoS/app/src/static/uploads/"
app.config["VIDEO_UPLOADS"] = "/mnt/c/Users/Eric/Desktop/side-projects/VideoS-unix/VideoS/app/src/static/uploads/"

@app.route('/', methods=['GET'])
def home():
    return render_template('upload.html')

@app.route('/upload-video', methods=['GET', 'POST'])
def uploadVideo():
    if request.method == "POST":
        if request.files:
            video = request.files['video']
            path = os.path.join(app.config["VIDEO_UPLOADS"], 'in.mp4')

            video.save(path)

            from src.scripts.video import stabilizeVideo
            stabilizeVideo(app.config["VIDEO_UPLOADS"], 'in.mp4')

            outputPath = os.path.join(app.config["VIDEO_UPLOADS"], 'out.mp4')
            redirect('/')
            return send_file(outputPath, as_attachment=True)


    return redirect('/')

@app.route('/test', methods=['GET'])
def test():
    return "hello world"
