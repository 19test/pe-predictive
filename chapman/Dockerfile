FROM continuumio/anaconda3

RUN apt-get update

# Install software dependencies
RUN pip install radnlp==0.2.0.8
RUN pip install seaborn==0.7.1

# Install jupyter notebook
RUN /opt/conda/bin/conda install jupyter -y --quiet 
RUN mkdir /opt/notebooks 

# Add the start script to the container root
ADD start.sh /
RUN chmod u+x /start.sh

# Add the notebooks
ADD ./notebooks /opt/notebooks/
WORKDIR /opt/notebooks

# Clean up
RUN apt-get autoremove -y
RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

CMD /start.sh
