import gym
import time
from threading import Lock
from pynput import keyboard

def run():
    env = gym.make('Tennis-v0')
    # ['NOOP', 'FIRE', 
    #  'UP', 'RIGHT',
    #  'LEFT', 'DOWN',
    #  'UPRIGHT', 'UPLEFT',
    #  'DOWNRIGHT', 'DOWNLEFT',
    #  'UPFIRE', 'RIGHTFIRE',
    #  'LEFTFIRE', 'DOWNFIRE',
    #  'UPRIGHTFIRE', 'UPLEFTFIRE',
    #  'DOWNRIGHTFIRE', 'DOWNLEFTFIRE']
    key_to_action = [
        's', 'S',
        'w', 'd',
        'a', 'x',
        'e', 'q',
        'c', 'z',
        'W', 'D',
        'A', 'X',
        'E', 'Q',
        'C', 'Z',
    ]
    modifiers = {'shift': False}
    lock = Lock()
    action_queue = []

    def on_release(key):
        if key == keyboard.Key.shift:
            modifiers['shift'] = False

    def on_press(key):
        if key == keyboard.Key.esc:
            raise KeyboardInterrupt
        if key == keyboard.Key.shift:
            modifiers['shift'] = True
        try:
            k = key.char  # single-char keys
            if modifiers['shift']:
                k = k.upper()
            with lock:
                action_queue.insert(0, k)
        except:
            pass

    env.reset()
    env.render('human')
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()  # start to listen on a separate thread
    while True:
        with lock:
            while len(action_queue):
                k = action_queue.pop()
                env.step(key_to_action.index(k))
                env.render('human')
        time.sleep(0.06)
            

if __name__ == '__main__':
    run()