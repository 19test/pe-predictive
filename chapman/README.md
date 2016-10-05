# Chapman Docker

This is a Docker image that will start a Jupyter notebook to run/show the Chapman analysis. Note that I have packaged all data and dependencies into this image, meaning for any urls to download data, the data is also included with this repo in the [notebooks/data/](notebooks/data/) folder.

Packages that need to be installed (e.g. seaborn and radnlp) have versions specified in case a future change breaks this code, you can see this in the top section of the [Dockerfile](Dockerfile).


## Getting Started
You should first [install Docker](https://docs.docker.com/engine/installation/) and then build the container:

    docker build -t vanessa/notebook .

You also need to install something called [Docker Compose](https://docs.docker.com/compose/install/), which is a nice way of putting containers together. Once you have done this, there is a simple command for starting the entire application.

    docker-compose up

"Up" means "bring it up." If we add the "-d" option this would detach it, but we actually don't want to do this so we can see the ip address to go to in our browser. 

If you need to restart or stop you can just press Control+C. If you use the detached option, then you would do:

      docker-compose restart notebook
      docker-compose stop notebook

If you want to see running images:

      docker ps

## Where is my notebook?

The notebook is going to be running on a server, and spit out it's IP address

      Recreating chapman_notebook_1
      Attaching to chapman_notebook_1
      notebook_1 | Starting notebook...
      notebook_1 | Open browser to 127.0.0.1:8888

## Running commands
When you click on an ipynb file, it will open up the interactive notebook. When you click on a cell and press Control+ENTER, this will execute the code in the cell. Note that when you first run the code, Matplotlib builds font libraries. This is annoying, but only will happen once.


## How do I shell into the container?
First you need to know the container id. If you do `docker ps` you will see it running as *_notebook_1. There is a weird 12 character code by its name, this is it's container-id `af21bf1d48a6` 

>> pro-tip: this is actually the first 12 characters of the image md5-sum! It's a much longer string, but we only need 12 to distinguish containers on a user local machine

Once we have the id, we can shell into it with the following command:

      docker exec -it af21bf1d48a6 bash

This says we want to execute (exec) and (interactive)(terminal) for container with id (af21bf1d48a6) and run the command (bash)


# Why?
By "shipping" analyses in packages, meaning having a specification of all dependencies (python modules, data, etc.) we can be assured that the next person that runs our analysis will not run into system-specific differences. They won't have to install python or anaconda to run our notebook, and get a weird message about having the wrong kernel. They just need Docker, and then to run the image, and that's it. This is an important feature of reproducible workflows and analyses, and every piece of code that you work on (and tend to share) should have features like this.
