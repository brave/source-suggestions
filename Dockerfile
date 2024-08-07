FROM public.ecr.aws/docker/library/python:3.12.5-slim-bullseye

RUN mkdir -p app
WORKDIR /app

COPY ./requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

COPY . /app/
