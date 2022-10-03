FROM nvidia/cuda:11.4.1-devel-ubuntu20.04

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute

RUN apt-get update \
  && apt-get install -y -qq --no-install-recommends \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    libgles2 \
    libxext6 \
    libx11-6 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /project

ADD https://github.com/deepmind/mujoco/releases/download/2.2.2/mujoco-2.2.2-linux-x86_64.tar.gz /project/

RUN tar -xzf /project/mujoco-2.2.2-linux-x86_64.tar.gz -C /project/

RUN ls /project/mujoco-2.2.2/lib

RUN apt-get update \
  && apt-get install -y -qq --no-install-recommends \
    libglfw3-dev \
  && rm -rf /var/lib/apt/lists/*

ENV PATH="${PATH}:/project/mujoco-2.2.2/lib"

COPY ./src .

RUN make

#CMD [ "/project/main", "/project/models/quad.xml"]

ENTRYPOINT [ "/project/mujoco-2.2.2/bin/simulate" ]

CMD [ "/project/models/two_wheel.xml"]