"""
跳起之后可以立刻衔接攻击，但是衔接格挡会有一定的延时，因此跳起时间应该设置较长
切换刃具之后可以立刻释放刃具
"""

from Utilities.key_set.directkeys import PressKey, ReleaseKey, dk
import time

def attack(Time=0.4):
    PressKey(dk['K'])
    time.sleep(Time)
    ReleaseKey(dk['K'])

def defense(Time=0.3):
    PressKey(dk['L'])
    time.sleep(Time)
    ReleaseKey(dk['L'])

def forward(Time=0.8):
    PressKey(dk['W'])
    time.sleep(Time)
    ReleaseKey(dk['W'])

def back(Time=0.8):
    PressKey(dk['S'])
    time.sleep(Time)
    ReleaseKey(dk['S'])

def left(Time=0.6):
    PressKey(dk['A'])
    time.sleep(Time)
    ReleaseKey(dk['A'])

def right(Time=0.6):
    PressKey(dk['D'])
    time.sleep(Time)
    ReleaseKey(dk['D'])

def tool(Time=0.01):
    PressKey(dk['Z'])
    time.sleep(Time)
    ReleaseKey(dk['Z'])

def usetool(Time=0.1):
    PressKey(dk['N'])
    time.sleep(Time)
    ReleaseKey(dk['N'])

def jump(Time=0.8):
    PressKey(dk["SPACE"])
    time.sleep(Time)
    ReleaseKey(dk["SPACE"])

def shift(Time=1):
    PressKey(dk["LSHIFT"])
    time.sleep(Time)
    ReleaseKey(dk["LSHIFT"])

def lock(Time=0.1):
    PressKey(dk['O'])
    time.sleep(Time)
    ReleaseKey(dk['O'])



if __name__ == '__main__':
    time.sleep(3)
    shift(1)
    attack()
