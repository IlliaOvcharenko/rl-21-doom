docker build --build-arg UID=$UID -t rl-21-doom-image .
docker run --name rl-21-doom-container \
           --gpus all --shm-size=11g \
           -dt -v $(pwd)/../:/home/dev/codebase/ \
           -p 5900:5900 \
           -w /home/dev/codebase \
           rl-21-doom-image

