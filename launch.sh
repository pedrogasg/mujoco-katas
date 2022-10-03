#/bin/zsh

cd $1

docker build -t $1 .

docker run --gpus all -it --rm --env="DISPLAY" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" --env="QT_X11_NO_MITSHM=1" -v $(pwd)/../models:/project/models $1:latest $2

cd ..