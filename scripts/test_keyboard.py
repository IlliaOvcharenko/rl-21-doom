from pynput import keyboard


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

    print(get_action())

