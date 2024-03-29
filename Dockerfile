FROM public.ecr.aws/docker/library/python:3.9.11-slim-bullseye

RUN mkdir -p app
WORKDIR /app

COPY ./requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

COPY . /app/
