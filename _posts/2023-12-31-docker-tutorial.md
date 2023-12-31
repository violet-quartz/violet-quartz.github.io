---
layout: post
title:  "Docker Tutorial"
date:   2023-12-31 15:32 +0800
categories: note
subtype: MLOps/DevOps
---

Docker is an open platform for developing, shipping, and running applications. It enables you to separate your applications from your infrastructure. With Docker, you can manage your infrastructure in the same ways you manage your applicaitons. Here we Introduce how to use docker and show its usage in ML/DL.

In Machine Learning/Deep Learning field, reproducing results are essential. Creating and developing ML models is often messy as there are different versions of the same software. With Docker, you can include the right versions of each needed dependency and library, so no one ever has to do any configuration. After the Dockerfile is built, you'll have exactly what you need. And it's easy for you to share the work.

# Table of Contents
- [Work with docker](#work-with-docker)
    - [Install docker](#install-docker)
    - [Concepts](#concepts)
    - [Commonly used CLI commands](#commonly-used-cli-commands)
- [Docker architecture](#docker-architecture)
- [Docker for machine learning](#docker-for-machine-learning)

## Work with docker 
### Install docker
- Install the latest version of [Docker Desktop](https://docs.docker.com/get-docker/).

- Double click to open Docker Desktop.

- Open cmd tool and verify installation.
    ```
    > docker info
    ```

### Concepts
**Container**: A container is an isolated environment for your code.

**Image**: an executable package that includes everything needed to run an applicaiton, like code, runtime, libraries, environment variables and configuration files.

**Docker file**: a text file that contains all commands, in order, needed to build a given image. Docker builds images automatically by reading the instructions from a dockerfile. [Docker file format and instructions](https://docs.docker.com/engine/reference/builder/).

**Docker Hub**: Docker registry. You can share, store, search and pull images in [Docker Hub](https://hub.docker.com/). 

### Commonly used CLI commands
- Manage image
```
> docker image ls -a
> docker image rm <image id>
> docker image rm $(docker image ls -a -q)
```

- Build image  
```
> cd directory_of_Dockerfile
> docker build -t your_name:your_tag PATH
```

- Pull image from Docker Hub
```
> docker pull hello-world 
```

- Upload image to Docker Hub
```
> docker tag image_id username_of_dockerhub/image_name:image_tag
> docker login
> docker push username_of_dockerhub/image_name:image_tag
```

- Run container
```
> docker container run image_name # Bring up the image and then shut it down again.
> docker container run -it image_name # Allow you to interact with the image
> docker run -it -v "%cd%":/data image_name bash
> docker run -it -v "%cd%":/data -p 8888:8888 image_name
```

- Stop/Manage containers
```
> docker container ls
> docker container stop <container id> # gracefully stop the container
> docker container kill <container id> # force shutdown the container
> docker container rm <container id>
```

## Docker architecture
Docker uses a client-server architecture. The Docker client talks to the Docker daemon, which does the heavy lifting of building, running, and distributing your Docker containers. The Docker client and daemon can run on the same system, or you can connect a Docker client to a remote Docker daemon. The Docker client and daemon communicate using a REST API, over UNIX sockets or a network interface. Another Docker client is Docker Compose, that lets you work with applications consisting of a set of containers.

{% include image.html post=page.path file="docker-architecture.jpg" alt="Docker architecture from doc.docker.com" width="600" %}

## Docker for Machine Learning
An example Dockerfile of setting machine learning development environment.

```Dockerfile
FROM continuumio/miniconda3

LABEL maintainer="violet-quartz"
LABEL description="Dev environment for machine learning and data science team \
                  with access to shared data in network drive."

WORKDIR /data

RUN conda install jupyter -y && \
    conda install python==3.7.3 -y && \
    conda install pandas==0.24.2 -y && \
    conda install seaborn==0.9.0 -y && \
    conda clean -y -all

EXPOSE 8888

VOLUME /data

CMD ["jupyter","notebook","--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

```




