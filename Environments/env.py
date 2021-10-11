"""
游戏窗口的位置为(8, 0, 1450, 849)，每个step为节省时间只进行一次桌面抓取（整个游戏窗口）
血量计算：测试苇名弦一郎像素在67-78之间，可能会有部分杂点，忽略不计，截取像素位置在(91, 106, 412, 145)，选取第五行为参考血量
相对位置为(83, 106, 404, 145)
测试狼像素在90-100之间，截取像素位置在(90, 765, 568, 804)，选取第十行为参考血量
相对位置为(82, 765, 560, 804)
截取游戏画面的像素位置为(454, 94, 988, 794)
相对位置为(446, 94, 980, 794)
狼满血为455
苇一郎满血251
"""

from Utilities.Get_Screen.get_screen import grab_screen, get_blood
import Run.action_set as action
import cv2
import time

def take_action(a):
    if a == 6:
        pass
    elif a == 0:
        action.forward()
    elif a == 1:
        action.back()
    elif a == 2:
        action.attack()
    elif a == 3:
        action.defense()
    elif a == 4:
        action.jump()
    elif a == 5:
        action.shift()
    elif a == 7:
        action.tool()
    elif a == 8:
        action.usetool()

def step(a, self_blood_last, enemy_blood_last, state):
    take_action(a)
    img = grab_screen((8, 0, 1450, 849))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    self_blood = get_blood(img[765:805, 82:561], 10, 90, 100) / 4.55  #(82, 765, 560, 804)
    enemy_blood = get_blood(img[106:146, 83:405], 5, 65, 77) / 2.51 #(83, 106, 404, 145)
    state_next = img[94:794, 446:980]                                #(446, 94, 980, 794)
    state.append(state_next)
    done = False
    if self_blood < 1:
        done = True
    reward = (self_blood - self_blood_last) * 0.3 + (enemy_blood_last - enemy_blood) * 5
    return reward, state, done, self_blood, enemy_blood

def restart():
    action.attack()
    print("restart")
    action.lock()
