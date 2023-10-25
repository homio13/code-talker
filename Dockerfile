FROM python:3.11.1-bullseye

ENV PORT 8080
ENV HOSTNAME 0.0.0.0

RUN mkdir /myapp
WORKDIR /myapp

RUN apt-get update
RUN pip install --upgrade pip

COPY . ./

RUN python -m pip install -r requirements.txt

CMD ["python", "app.py"]

EXPOSE 8080