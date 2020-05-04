from flask import Flask, request, redirect, jsonify, render_template
import os


# init
app = Flask(__name__)
app.config["VIDEO_UPLOADS"] = "C:/Users/Eric/Desktop/side projects/VideoS/app/src/static/uploads"


@app.route('/', methods=['GET'])
def home():
    return render_template('upload.html')#return jsonify({'msg' : "hello world"})

@app.route('/upload-video', methods=['GET', 'POST'])
def uploadVideo():
    if request.method == "POST":
        if request.files:
            video = request.files['video']
            path = os.path.join(app.config["VIDEO_UPLOADS"], video.filename)

            video.save(path)

            from scripts.video import stabilizeVideo
            stabilizeVideo(app.config["VIDEO_UPLOADS"], video.filename)
            #print(video)

            return redirect(request.url)


    return redirect('/')


# start server
if __name__ == "__main__":
    app.run(debug=True, threaded=True)
