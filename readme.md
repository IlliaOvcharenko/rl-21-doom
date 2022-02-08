# 👾 Doom

<p align="center">
<img width="320" height="240" align="center" src="figs/deadly_corridor-exmpl.gif"/>
<img width="320" height="240" align="center" src="figs/basic-exmpl.gif"/>
</p>

## How to run?
Build and run docker container.
```
cd docker
./run_docker.sh
./attach_docker.sh
```


Run virtual screen buffer in you want to meditate how your model trains via VNC.
```
./run_virtual_display.sh

```

Run training!

```
pyhton scripts/run_train.py --n_epoches=10 --n_episode_to_play=10 --game_scenario="deadly_corridor" --encoder="effnet"

```

