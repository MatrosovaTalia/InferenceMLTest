FROM python:3.8

WORKDIR /work
RUN apt-get update && apt-get -y install cmake
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
ENTRYPOINT ["python", "people_tracker.py"]
CMD ["-i", "test.mp4"]

