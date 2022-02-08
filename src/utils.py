from pathlib import Path

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


def save_episodes(episodes_info, model_full_name, figs_folder):
    episodes_folder = figs_folder / model_full_name
    episodes_folder.mkdir(parents=True, exist_ok=True)
    for episode_idx in range(len(episodes_info)):
        reward = episodes_info[episode_idx]["reward"]
        frames = episodes_info[episode_idx]["frames"]
        sfn = episodes_folder / f"episode={episode_idx}-reward={reward:.4f}.gif"
        save_gif(sfn, frames, 10)


def get_next_model_version(base_name, models_folder, name_sep="-", version_prefix="v"):
    names = list(filter(lambda fn: fn.is_dir(), models_folder.glob("*")))
    names = [name.stem for name in names]
    names = [name for name in names if base_name in name]
    names = sorted(names)

    next_version = 0
    if len(names) > 0:
        last_version = names[-1].split(name_sep)[-1]
        last_version = last_version.replace(version_prefix, "")
        last_version = int(last_version)
        next_version = last_version+1
    return next_version


def compose_model_name(base_name, version, name_sep="-", version_prefix="v"):
    name = f"{base_name}{name_sep}{version_prefix}{version}"
    return name

