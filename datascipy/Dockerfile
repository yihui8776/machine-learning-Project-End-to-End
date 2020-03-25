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
RUN apk add --no-cache  --virtual sklearn-runtime make cmake    git bash zlib hdf5 libgfortran libgcc libstdc++ musl  gfortran g++ gcc  libquadmath libgfortran
RUN apk add --no-cache --virtual .build-deps build-base   zlib-dev hdf5-dev freetype-dev libpng-dev  musl-dev cmake  && \
    ln -s /usr/include/locale.h /usr/include/xlocale.h && \
        apk del .build-deps && \
        rm -rf /var/cache/apk/*

RUN apk add --update  py-pip  python3-dev --no-cache
 # pillow 依赖
RUN apk add jpeg-dev zlib-dev freetype-dev lcms2-dev openjpeg-dev tiff-dev tk-dev tcl-dev \
 # scipy 
 && apk add -f --no-cache openblas openblas-dev lapack  lapack-dev --no-cache


# 根据需要增加Python库
COPY requirements.txt /requirements.txt
#python 库
ENV NUMPY_VER 1.15.3
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

RUN git clone https://github.com/scipy/scipy.git  --depth=1 && cd scipy && git clean -xdf && python setup.py install --user

RUN pip3 install -i https://pypi.douban.com/simple/ \ 
         --trusted-host  pypi.douban.com  \
         --no-cache-dir  \
         -r /requirements.txt


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

#tensoflow
#RUN pip3 install tensorflow --ignore-installed numpy

# TensorBoard
#EXPOSE 6006
# IPython
EXPOSE 8888
# SSH
EXPOSE 22

WORKDIR "/notebooks"

CMD ["/run_jupyter.sh", "--allow-root"]

