## **Sample FastAPI Deployment for Machine Learning Models**

This project demonstrates how to deploy a machine learning model using FastAPI and Docker. The example uses the Iris dataset for training a RandomForest model and serves predictions via a FastAPI application.

Note you'll need Docker desktop installed.

```train.py``` trains a simple model and saves the trained model as a .pkl file.

```app.py``` contains the application code which exposes an endpoint to the trained model and can make predictions on the classic Iris dataset.

You can test the app locally by running: ```uvicorn app:app --reload``` (reload means code changes are reflected in the app -- don't use in prod)

Once happy, and you have Docker desktop installed, cd to your app directory and run:

```docker build -t fastapi-iris-app .```

Once the container is built, run:

```docker run -p 8000:8000 fastapi-iris-app```

This will launch and run the Dockerised version of your app, in this case at:

http://127.0.0.1:8000

To use the dockerfile and edit interactively:

```docker -compose --build```

To push to docker hub, first login:

```docker login```

Now we can tag the image and push it to Docker hub. You need your Docker username here.

```docker tag fastapi-iris-app:latest swordsjoshua91637/fastapi-iris-app:latest```

```docker push swordsjoshua91637/fastapi-iris-app:latest```


To run this Docker image
```docker pull swordsjoshua91637/fastapi-iris-app:latest```

```docker run -d -p 8000:8000 swordsjoshua91637/fastapi-iris-app:latest```

go to the localhost to see the app running. You can now pul & run this Docker image from anywhere at anytime!

To stop, first see running:
```docker ps```

Get the ID (e.g. bd2efb4d8b3c) anr run:
```docker stop bd2efb4d8b3c```


**Next steps**:
- Deploy e.g. Google Cloud Run, Heroku (no free tier anymore), or even Kubernetes

* Github actions workflows. Automate test & build.

