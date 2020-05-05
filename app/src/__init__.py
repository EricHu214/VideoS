from flask import Flask, request, redirect, jsonify, render_template, send_file
import os


# init
app = Flask(__name__)
app.config["VIDEO_UPLOADS"] = "C:/Users/Eric/Desktop/side projects/VideoS/app/src/static/uploads/"


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

            from src.scripts.video import stabilizeVideo
            stabilizeVideo(app.config["VIDEO_UPLOADS"], video.filename)

            outputPath = os.path.join(app.config["VIDEO_UPLOADS"], 'out.mp4')
            print(outputPath)
            #return send_file(outputPath, 'video/mp4')
            html = '''
                    <!doctype html>
                    <html>
                        <head>
                            <title>Stabilized Video</title>
                        </head>
                        <body>
                            <video width="720" controls>
                                <source src="./src/static/uploads/out.mp4" type="video/mp4">
                            </video>

                            <a href="./src/static/uploads/out.mp4" download> Download </a>
                        </body>
                    </html>
                '''
            return html


    return redirect('/')

@app.route('/test', methods=['GET'])
def test():
    return "hello world"


# start server
# if __name__ == "__main__":
#     app.run(debug=True, threaded=True)
