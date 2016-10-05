#!/bin/sh
echo "Starting notebook..."
ipaddress=`awk 'NR==1 {print $1}' /etc/hosts`
echo "Open browser to $ipaddress:8888"
/opt/conda/bin/jupyter notebook --notebook-dir=/opt/notebooks --ip='*' --port=8888 --no-browser
