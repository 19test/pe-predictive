FROM ubuntu:15.10

MAINTAINER Vanessa Sochat <vsochat@stanford.edu>

RUN apt-get update && apt-get install -y \
    cython3 \
    gcc \
    ipython3-notebook \
    mc \
    nano \
    python3 \
    python3-numpy \
    python3-pip \
    python3-setuptools \
    python3-scipy \
    vim

RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8 
ENV LC_CTYPE en_US.UTF-8
ENV LC_ALL en_US.UTF-8
ENV CODE_HOME /code
RUN easy_install3 --upgrade gensim
RUN apt-get install python3-pandas
RUN apt-get install python3-numpy
RUN apt-get install python3-scipy
RUN pip3 install scikit-learn
RUN pip3 install wordfish
RUN python3 -c "import nltk; nltk.download('all')"

RUN alias python='python3'
RUN alias ipython='ipython3'
RUN alias pip='pip3'
RUN mkdir /code
WORKDIR /code
ADD . /code/

EXPOSE 9000

CMD ["/bin/bash"]
