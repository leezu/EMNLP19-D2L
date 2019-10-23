FROM 763104351884.dkr.ecr.us-west-2.amazonaws.com/mxnet-inference:1.4.1-gpu-py3
# If running outside of us-west-2, change us-west-2 in above URL to the region you're running from.

RUN pip install --upgrade --user --pre 'mxnet-cu100==1.6.0b20191101' 'git+https://github.com/dmlc/gluon-nlp.git#egg=gluonnlp[extras]'

COPY *.py /opt/ml/model/code/