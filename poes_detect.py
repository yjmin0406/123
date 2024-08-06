# encoding=utf-8
import math
import numpy as np


def calculate_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance


def get_angle(p1, p2, p3):
    v1 = p1 - p2
    v2 = p3 - p2
    angle = np.dot(v1, v2) / (np.sqrt(np.sum(v1 * v1)) * np.sqrt(np.sum(v2 * v2)))
    angle = np.arccos(angle) / 3.14 * 180
    cross = v2[0] * v1[1] - v2[1] * v1[0]
    if cross < 0:
        angle = - angle
    return abs(angle)


def aspectRatio(box):
    box_list = box.tolist()
    h = box_list[3]-box_list[1]
    w = box_list[2]-box_list[0]
    bi = w / h
    return [bi, w, h]


def is_fallen(keypoints, boxes):
    keypoints = np.array(keypoints)

    # 自定义阈值
    ATHERPOSE = 0
    HUMAN_ANGLE = 25
    ASPECT_RATIO = 0.6
    Vertical_threshold = 0
    Horizontal_threshold = 0

    Left_Shoulder = keypoints[5]
    if Left_Shoulder[0] + Left_Shoulder[1] == 0:
        ATHERPOSE += 1
    Right_Shoulder = keypoints[6]
    if Right_Shoulder[0] + Right_Shoulder[1] == 0:
        ATHERPOSE += 1
    Left_Hip = keypoints[11]
    if Left_Hip[0] + Left_Hip[1] == 0:
        ATHERPOSE += 1
    Right_Hip = keypoints[12]
    if Right_Hip[0] + Right_Hip[1] == 0:
        ATHERPOSE += 1
    Left_Knee = keypoints[13]
    if Left_Knee[0] + Left_Knee[1] == 0:
        ATHERPOSE += 1
    Right_Knee = keypoints[14]
    if Right_Knee[0] + Right_Knee[1] == 0:
        ATHERPOSE += 1
    Left_Ankle = keypoints[15]
    if Left_Ankle[0] + Left_Ankle[1] == 0:
        ATHERPOSE += 1
    Right_Ankle = keypoints[16]
    if Right_Ankle[0] + Right_Ankle[1] == 0:
        ATHERPOSE += 1

    # 计算左右肩膀的中心点
    Shoulders_c = np.array([(Left_Shoulder[0] + Right_Shoulder[0]) // 2, (Left_Shoulder[1] + Right_Shoulder[1]) // 2])
    ''''计算左右胯部的中心点'''
    hips_c = np.array([(Left_Hip[0] + Right_Hip[0]) // 2,
              (Left_Hip[1] + Right_Hip[1]) // 2])
    '''计算左右膝盖的中心点'''
    Knee_c = np.array([(Left_Knee[0] + Right_Knee[0]) // 2,
              (Left_Knee[1] + Right_Knee[1]) // 2])
    '''计算左右脚踝的中心点'''
    Ankle_c = np.array([(Left_Ankle[0] + Right_Ankle[0]) // 2,
               (Left_Ankle[1] + Right_Ankle[1]) // 2])
    '''计算身体中心线与水平线夹角'''
    human_angle = get_angle(Shoulders_c, hips_c, np.array([0, hips_c[1]]))
    '''计算检测区域宽高比'''
    aspect_ratio = aspectRatio(boxes)
    # print("宽高比::", aspect_ratio)
    '''计算肩部中心点与胯部中心点的垂直距离差'''
    human_shoulderhip = abs(Shoulders_c[1] - hips_c[1])

    '''计算肩部胯部膝盖夹角'''
    Hip_Knee_Shoulders_angle = get_angle(Shoulders_c, hips_c, Knee_c)
    Hip_Knee_Right_angle = get_angle(Right_Shoulder, Right_Hip, Right_Knee)
    # print(Hip_Knee_Shoulders_angle, Hip_Knee_Right_angle)

    '''计算胯部膝盖小腿夹角'''
    Ankle_Knee_Hip_angle = get_angle(hips_c, Knee_c, Ankle_c)
    Ankle_Knee_Right_angle = get_angle(Right_Hip, Right_Knee, Right_Ankle)

    '''计算胯部膝盖是否处于相似的垂直位置'''
    vertical_threshold = (Left_Knee[1] - Left_Shoulder[1])/aspect_ratio[-1]

    '''计算胯部膝盖是否处于相似的水平位置'''
    horizontal_threshold = (Left_Shoulder[0] - Left_Knee[0]) / aspect_ratio[1]
    '''初始化站、坐、摔倒的指数值'''
    status_score = {'Stand': 0.0,
                    'Fall': 0.0,
                    'Sit': 0.0,
                    'other': 0.0}
    _weight = ''

    if ATHERPOSE >= 8:
        status_score['other'] += 5.6

    '''判断Shoulder、Hip、Knee是否被检测到'''
    if Knee_c[0] == 0 and Knee_c[1] == 0 and hips_c[0] == 0 and hips_c[1] == 0:
        status_score['Sit'] += 0.69
        status_score['Fall'] += -0.8 * 2
        status_score['Stand'] += -0.8 * 2
        _weight = f'[1]Sit:+0.2, Fall:-1.6 ,Stand: -1.6'

    elif Shoulders_c[1] == 0 and Shoulders_c[0] == 0 and hips_c[0] == 0 and hips_c[1] == 0:
        status_score['Sit'] += -0.8 * 2
        status_score['Fall'] += -0.8 * 2
        status_score['Stand'] += 0.69
    else:
        if 180 > Ankle_Knee_Hip_angle > 125 and aspect_ratio[0] < ASPECT_RATIO:
            status_score['Stand'] += 1.6
        if 25 > Ankle_Knee_Hip_angle > -25 and aspect_ratio[0] > 1 / ASPECT_RATIO:
            status_score['Fall'] += 1.6
        if 125 > Ankle_Knee_Hip_angle > 75:
            status_score['Sit'] += 0.6
        _weight = f'[1]Sit:+0.2, Fall:-1.6 ,Stand: -1.6'

    '''身体中心线与水平线夹角+-25'''
    if human_angle in range(-HUMAN_ANGLE, HUMAN_ANGLE):
        status_score['Fall'] += 0.8
        status_score['Sit'] += 0.1
        _weight = f'{_weight}, [2]Fall:+0.8, Sit:+0.1'
    else:
        status_score['Fall'] += 0.2 * ((90 - human_angle) / 90)
        _weight = f'{_weight}, [3]Fall:+{0.8 * ((90 - human_angle) / 90)}'

    '''宽高比小与0.6则为站立'''
    if aspect_ratio[0] < ASPECT_RATIO and human_angle in range(65, 115):
        status_score['Stand'] += 0.8
        _weight = f'{_weight}, [4]Stand:+0.8'

    elif aspect_ratio[0] > 1 / ASPECT_RATIO:  # 5/3
        status_score['Fall'] += 0.8
        _weight = f'{_weight}, [5]Fall:+0.8'

    if vertical_threshold < Vertical_threshold:
        status_score['Fall'] += 0.6
        status_score['Sit'] += -0.15

    if horizontal_threshold < Horizontal_threshold:
        status_score['Fall'] += 0.6
        status_score['Sit'] += -0.15

    if 25 < Hip_Knee_Shoulders_angle < 145 and 75 < human_angle < 125:
        status_score['Sit'] += 0.8
        status_score['Stand'] += -0.035
        if vertical_threshold > Vertical_threshold:
            status_score['Sit'] += +0.15
        _weight = f'{_weight}, [6]Stand:-0.035, Sit:+0.15'

    elif Hip_Knee_Shoulders_angle > 120 and 75 < human_angle < 125:
        status_score['Stand'] += 0.2

    elif Hip_Knee_Shoulders_angle > 120 and -25 < human_angle < 25:
        status_score['Fall'] += 0.2

    else:
        status_score['Fall'] += 0.05
        status_score['Stand'] += 0.05
        _weight = f'{_weight}, [7]Stand:+0.05, Fall:+0.05'

    if 25 < Ankle_Knee_Hip_angle < 145 and 45 < human_angle < 125:
        status_score['Sit'] += 0.8
        status_score['Stand'] += -0.035
        if vertical_threshold > Vertical_threshold:
            status_score['Sit'] += +0.15
        _weight = f'{_weight}, [8]Stand:+0.035, Sit:+0.15'
    else:
        status_score['Fall'] += 0.05
        status_score['Stand'] += 0.05
        _weight = f'{_weight}, [9]Stand:+0.05, Sit:+0.05'

    if 65 < Hip_Knee_Right_angle < 145 and 45 < human_angle < 125:
        status_score['Sit'] += 0.8
        if vertical_threshold > Vertical_threshold:
            status_score['Sit'] += +0.15
    else:
        status_score['Fall'] += 0.05
        status_score['Stand'] += 0.05
    score_max, status_max = max(zip(status_score.values(), status_score.keys()))

    if status_max == 'Stand' or status_max == 'Sit' or status_max == 'other':
        status_max = 'Normal'
    else:
        status_max = 'Dangerous'

    return status_max
