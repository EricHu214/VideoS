from ubuntu:latest

run apt-get update && apt-get install -y software-properties-common
run add-apt-repository ppa:deadsnakes/ppa
run apt-get install -y python3-dev
run apt install -y python3-pip
run pip3 install --upgrade pip

workdir /
copy . /

run python3 app/setup.py install

expose 8080

entrypoint gunicorn -t 600 -b :8080 main:app
