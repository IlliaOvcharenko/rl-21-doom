import os

import vizdoom as vzd
from pynput import keyboard

from random import choice
from time import sleep


def get_action():
    action_map = {
        "'w'": [0, 0, 1, 0, 0],
        "'a'": [1, 0, 0, 0, 0],
        "'s'": [0, 0, 0, 1, 0],
        "'d'": [0, 1, 0, 0, 0],
        "Key.space": [0, 0, 0, 0, 1],
    }

    action = None

    def on_press(k):
        nonlocal action
        if str(k) in action_map.keys():
            action = action_map[str(k)]
            return False

    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()

    return action


if __name__ == "__main__":

    game = vzd.DoomGame()
    game.set_doom_scenario_path(os.path.join(vzd.scenarios_path, "basic.wad"))

    game.set_doom_map("map01")
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)

    game.set_screen_format(vzd.ScreenFormat.RGB24)


    game.set_depth_buffer_enabled(True)
    game.set_labels_buffer_enabled(True)
    game.set_automap_buffer_enabled(True)
    game.set_objects_info_enabled(True)
    game.set_sectors_info_enabled(True)

    game.set_render_hud(False)
    game.set_render_minimal_hud(False)
    game.set_render_crosshair(False)
    game.set_render_weapon(True)
    game.set_render_decals(False)
    game.set_render_particles(False)
    game.set_render_effects_sprites(False)
    game.set_render_messages(False)
    game.set_render_corpses(False)
    game.set_render_screen_flashes(True)

    game.set_available_buttons([
        vzd.Button.MOVE_LEFT,
        vzd.Button.MOVE_RIGHT,
        vzd.Button.MOVE_FORWARD,
        vzd.Button.MOVE_BACKWARD,
        vzd.Button.ATTACK,
    ])

    game.set_available_game_variables([vzd.GameVariable.AMMO2])

    game.set_episode_timeout(200)
    game.set_episode_start_time(10)
    game.set_window_visible(True)
    game.set_living_reward(-1)
    game.set_mode(vzd.Mode.PLAYER)

    game.init()


    episodes = 10

    for i in range(episodes):
        print("Episode #" + str(i + 1))

        game.new_episode()

        while not game.is_episode_finished():

            state = game.get_state()

            n = state.number
            vars = state.game_variables
            screen_buf = state.screen_buffer
            depth_buf = state.depth_buffer
            labels_buf = state.labels_buffer
            automap_buf = state.automap_buffer
            labels = state.labels
            objects = state.objects
            sectors = state.sectors

            r = game.make_action(get_action())


        print("Episode finished.")
        print("Total reward:", game.get_total_reward())
        print("************************")

    game.close()

