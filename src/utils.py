from src.frameio import (save_gif,
                         save_video)


def record(model, agent):
    frames = list(agent.cur_frames)
    epsilon = 0.2
    done = False
    while not done:
        reward, done = agent.play_step(model, epsilon)
        frames += list(agent.cur_frames)

    save_gif("test.gif", frames, 10)
    save_video("test.mp4", frames, 10)

