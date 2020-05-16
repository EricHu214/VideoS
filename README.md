## VideoS
A Flask Based video stabilizer to upload a video and have it stabilized. This is a loose implementation of video stabilization based on the MeshFlow paper. This is a simpler approach with much less computation compared to the method described in the paper, but still delivers good results. Since this is a 2D feature point tracking implementation, this will produce subpar results on videos with many moving objects or videos with little to no distinct features (e.g a flat wall)

## How to run this development server
- clone repo
- setup a virtual environment (something like python venv)
- "python3 setup.py install" to install all the dependencies in the virtual environment
- "flask run" 
- navigate to "localhost:5000" to use

## Reference
Ported from: 
[Video_Stabilizer](https://github.com/EricHu214/Video_Stabilizer)
