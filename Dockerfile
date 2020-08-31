FROM tiangolo/uwsgi-nginx-flask:python3.7
RUN apt-get update -y
RUN apt-get install -y python-pip python-dev build-essential
WORKDIR /gender-detector
ADD . /gender-detector
RUN pip install -r requirements.txt
CMD ["python","app.py"]