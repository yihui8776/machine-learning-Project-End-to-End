FROM python:3.7-alpine
MAINTAINER yihui8776 <wangyaohui8776@sina.com>

#alpine国内镜像
RUN echo "http://mirrors.aliyun.com/alpine/v3.9/main/" > /etc/apk/repositories

#安装依赖
RUN apk update
RUN apk upgrade
RUN apk add --no-cache  gcc openssl-dev openssl libressl libc-dev linux-headers libffi-dev libxml2-dev libxml2 libxslt-dev g++


#jupyter
RUN pip3 install jupyter  && ln -s /usr/local/python37/bin/jupyter /usr/bin/jupyter

# SSH Server
# 安装openssh 并修改配置文件和生成key 并且同步时间
RUN apk add --no-cache openssh tzdata && \ 
    cp /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && \
    sed -i "s/#PermitRootLogin.*/PermitRootLogin yes/g" /etc/ssh/sshd_config && \
    ssh-keygen -t dsa -P "" -f /etc/ssh/ssh_host_dsa_key && \
    ssh-keygen -t rsa -P "" -f /etc/ssh/ssh_host_rsa_key && \
    ssh-keygen -t ecdsa -P "" -f /etc/ssh/ssh_host_ecdsa_key && \
    ssh-keygen -t ed25519 -P "" -f /etc/ssh/ssh_host_ed25519_key && \
    echo "root:ai1234" > /tmp/passwd && \
    chpasswd < /tmp/passwd && \
    rm -rf /tmp/passwd


#RUN sed -i 's/^PermitEmptyPasswords .*/PermitEmptyPasswords yes/g' /etc/ssh/sshd_config && \


# Set up our notebook config.
COPY jupyter_notebook_config.py /root/.jupyter/

# Copy sample notebooks.
COPY notebooks /notebooks

# Jupyter has issues with being run directly:
# We just add a little wrapper script.
COPY run_jupyter.sh /
RUN chmod +x  /run_jupyter.sh
RUN chmod +x /notebooks

# IPython
EXPOSE 8888
# SSH
EXPOSE 22

WORKDIR "/notebooks"

CMD ["/run_jupyter.sh", "--allow-root"]        
