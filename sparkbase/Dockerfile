FROM yihui8776/jupyter-alpine:v0.1
MAINTAINER Wangyaohui <wangyaohui8776@sina.com>
LABEL version="1.0"

USER root 
#alpine镜像
RUN echo "http://mirrors.aliyun.com/alpine/v3.11/main/" > /etc/apk/repositories;\
    echo "http://mirrors.aliyun.com/alpine/v3.11/community" >> /etc/apk/repositories; \
    echo "http://dl-cdn.alpinelinux.org/alpine/edge/testing" >> /etc/apk/repositories; 

#更新apk
RUN apk update
RUN apk upgrade


#基础依赖
RUN apk add --no-cache  --virtual sklearn-runtime make cmake git bash zlib hdf5 libgfortran libgcc libstdc++ musl  gfortran g++ gcc  libquadmath libgfortran  
RUN apk add --no-cache --virtual .build-deps build-base   zlib-dev hdf5-dev freetype-dev libpng-dev  musl-dev zeromq-dev  && \
    ln -s /usr/include/locale.h /usr/include/xlocale.h && \
        apk del .build-deps && \
        rm -rf /var/cache/apk/*

#java
RUN apk --update add curl openjdk8-jre-base && \
      rm -rf /var/cache/apk/*

# Set environment
ENV JAVA_HOME /usr/lib/jvm/default-jvm
ENV PATH ${PATH}:${JAVA_HOME}/bin


#python
RUN apk add --update  python3 py-pip  python3-dev --no-cache
 # pillow 依赖
RUN apk add jpeg-dev zlib-dev freetype-dev lcms2-dev openjpeg-dev tiff-dev tk-dev tcl-dev \
 # scipy  
 && apk add -f --no-cache openblas openblas-dev lapack  lapack-dev --no-cache




# 根据需要增加Python库
COPY requirements.txt /requirements.txt
#python 库
ENV NUMPY_VER 1.16.5
ENV SCIPY_VER 1.3.1

#update
RUN pip3 install --upgrade pip -i https://pypi.douban.com/simple


#numpy
RUN pip3 install numpy==$NUMPY_VER  -i https://pypi.douban.com/simple --no-cache-dir


#scipy  need pybind11 cython
RUN pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple/  pybind11 --no-cache-dir
 
RUN pip3 install Cython --install-option="--no-cython-compile" -i https://pypi.douban.com/simple --no-cache-dir

#scipy
RUN git config  --global   http.sslVerify "false" 
RUN git config --global http.postBuffer 1024288000

RUN git clone https://github.com/scipy/scipy.git  --depth=1 && cd scipy && git clean -xdf && python3 setup.py install --user

RUN pip3 install -i https://pypi.douban.com/simple/ \ 
         --trusted-host  pypi.douban.com  \
         --no-cache-dir  \
         -r /requirements.txt


#clean cache
#RUN rm -rf /tmp/*  && \
#   rm -rf /var/cache/apk/* && \
#   rm -rf /scipy 



#sparkbase
ENV ENABLE_INIT_DAEMON true
ENV INIT_DAEMON_BASE_URI http://identifier/init-daemon
ENV INIT_DAEMON_STEP spark_master_init

ENV SPARK_VERSION=2.4.5
ENV HADOOP_VERSION=2.7


RUN  cd /   && wget https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz \ 
       &&   tar -xvzf spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz \ 
       &&  mv spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION} spark \ 
       &&   rm spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz \ 
       && cd /


# Spark config
ENV SPARK_HOME /spark
ENV PYTHONPATH $SPARK_HOME/python:$SPARK_HOME/python/lib/py4j-0.10.7-src.zip
ENV SPARK_OPTS --driver-java-options=-Xms1024M --driver-java-options=-Xmx2048M --driver-java-options=-Dlog4j.logLevel=info

# Fix the value of PYTHONHASHSEED
# Note: this is needed when you use Python 3.3 or greater
ENV PYTHONHASHSEED 1
ENV  PYSPARK_ALLOW_INSECURE_GATEWAY 1


#jupyter notebook
RUN pip install  jupyter -i https://pypi.douban.com/simple/ 
RUN pip3 install  --no-cache-dir jupyter_contrib_nbextensions -i  https://pypi.douban.com/simple/ && jupyter contrib nbextension install --user

#kernel
RUN wget https://downloads.lightbend.com/scala/2.11.12/scala-2.11.12.tgz \
    && mkdir /usr/lib/scala \
    && tar -zxvf scala-2.11.12.tgz -C /usr/lib/scala/
ENV SCALA_HOME=/usr/lib/scala/scala-2.11.12
ENV PATH=$SCALA_HOME/bin:$PATH

RUN cd /
 RUN   git clone https://gitee.com/tann/jupyter-scala.git && \
    cd jupyter-scala && ./jupyter-scala

RUN pip3 --no-cache-dir install toree==0.2.0 -i  https://pypi.douban.com/simple/ 
RUN jupyter toree install --interpreters=Scala --spark_home=$SPARK_HOME --user --kernel_name=apache_toree --interpreters=PySpark,SparkR,Scala,SQL

#clean cache
RUN rm -rf /tmp/*  && \
   rm -rf /var/cache/apk/* && \
   rm -rf /scipy

# 重新notebook配置
COPY jupyter_notebook_config.py /root/.jupyter/
#
# 拷贝 sample notebooks.
COPY notebooks /notebooks
#
# Jupyter运行脚本
COPY run_jupyter.sh /
RUN chmod +x  /run_jupyter.sh
RUN chmod +x /notebooks

# IPython
EXPOSE 8888
# SSH
EXPOSE 22


WORKDIR "/notebooks"

#CMD ["/run_jupyter.sh", "--allow-root"]

