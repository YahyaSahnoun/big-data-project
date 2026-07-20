FROM spark-custom

USER root

RUN apt-get update && apt-get install -y python3-pip

RUN pip3 install \
    notebook \
    jupyterlab \
    pandas \
    numpy \
    matplotlib \
    scikit-learn \
    pyarrow \
    boto3 \
    pyspark==3.5.1

WORKDIR /workspace

CMD ["jupyter","lab","--ip=0.0.0.0","--port=8888","--no-browser","--allow-root","--NotebookApp.token="]