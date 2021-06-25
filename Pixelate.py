import numpy as np
import cv2
import pix_main_arena
import time
import gym
import pybullet as p
import pybullet_data
import cv2
import numpy as np
import cv2.aruco as aruco
import math
import os

import sys

curr=143

# function for getting a matrix which will be used for adjacency

ARUCO_PARAMETERS = aruco.DetectorParameters_create()
ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)


# Create grid board object we're using in our stream
board = aruco.GridBoard_create(
        markersX=2,
        markersY=2,
        markerLength=0.09,
        markerSeparation=0.01,
        dictionary=ARUCO_DICT)

def angle(vector_1, vector_2):
    return (np.angle(complex(vector_2[0], vector_2[1]) / complex(vector_1[0], vector_1[1])) * 180) / math.pi
def Euclidean_Distance(coordinate_1, coordinate_2):
    return math.sqrt((coordinate_1[0] - coordinate_2[0]) ** 2 + (coordinate_1[1] - coordinate_2[1]) ** 2)

def image_coordinates(a,n):
    y,x =a // n , a % n;
    print("-->",x,y,"<--")
    return np.array([(x*53)+67, (y*53)+64], dtype = np.int)
# north

def arucovector(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect Aruco markers
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMETERS)
    print("kyaa")
    print(corners)

    x = 0
    y = 0
    for arr in corners[0][0]:
        if(arr[0]>460):
            arr[0]-=(5*arr[0])/100
        elif(arr[0]<360):
            arr[0] += (5 * arr[0]) / 100
        if (arr[1] > 460):
            arr[1] -= (5 * arr[1]) / 100
        elif(arr[1]< 360):
            arr[1] += (5 * arr[1]) / 100
        x = x + arr[0]
        y = y + arr[1]

    arucoinitial = [x / 4, y / 4]
    print("ayaya")
    print(arucoinitial)
    arr = corners[0][0]
    vec1 = [arr[0][1] - arr[3][1], arr[0][0] - arr[3][0]]
    return arucoinitial, vec1


# def Grid_Coordinate(coordinate):
#     return np.array([(coordinate[1] - thickness[1]) / (size[1] / n_rows),
#                      (coordinate[0] - thickness[0]) / (size[0] / n_cols)], dtype=np.int)


def Move_Bot(factor, move):
    if move == "F" or move == "B":
        speed = int(min(20, max(factor - 50, 10), max(factor - 30, 5)))
        # speed=2
        if move == "F":
            env.move_husky(speed, speed, speed, speed)
        elif move == "B":
            env.move_husky(-speed, -speed, -speed, -speed)

        for _ in range(int(min(10, factor - 10))):
            p.stepSimulation()

    elif move == "L" or move == "R":
        speed = int(min(20, factor - 5))

        if move == "L":
            env.move_husky(-speed, speed, -speed, speed)
        elif move == "R":
            env.move_husky(speed, -speed, speed, -speed)

        for _ in range(int(min(10, factor))):
            p.stepSimulation()

    env.move_husky(0, 0, 0, 0)
    p.stepSimulation()


def letsgohusky(list,n):
    points=len(list)
    for i in range(points):
        destination = image_coordinates(list[i],n)
        while True:
            img = env.camera_feed()
            position, bot_vector= arucovector(img)
            # print("shape hai bhai -->",img.shape)
            distance = Euclidean_Distance(position, destination)
            # print("---->")
            # print([position, destination])
            if distance > 12:
                my_vector=np.array([ destination[1] - position[1],destination[0] - position[0]], dtype=np.int)
                theta = angle(bot_vector, my_vector)
                print("dest",destination)
                print("pos",position)
                print("my,bot",my_vector,bot_vector)
                print("theta",theta)
                # print("---->")
                # print(bot_vector,my_vector)
                # print("---->")
                # print(theta,list[i])
                theta=0-theta

                if theta <= 10 and theta >= -10:
                    Move_Bot(distance, "F")
                elif theta < -5 and theta > -125:
                    Move_Bot(-theta, "L")
                elif theta > 5 and theta < 125:
                    Move_Bot(theta, "R")
                elif theta >= 170 or theta <= -170:
                    Move_Bot(distance, "B")
                elif theta >= 125 and theta < 175:
                    Move_Bot(180 - theta, "L")
                elif theta <= -125 and theta > -175:
                    Move_Bot(180 + theta, "R")
            else:
                break




def forsquares(res, imgtest, lower, upper, d):
    imghsv = cv2.cvtColor(imgtest, cv2.COLOR_BGR2HSV)
    # imghsv= imgtest
    mask = cv2.inRange(imghsv, lower, upper)
    lower2 = np.array([170, 70, 50])
    upper2 = np.array([180, 255, 255])
    masked1 = cv2.inRange(imghsv, lower, upper)
    masked2 = cv2.inRange(imghsv, lower2, upper2)
    mask_red = masked1 + masked2
    if (d == 4):
        mask = mask_red
    imgresult = cv2.bitwise_and(imgtest, imgtest, mask=mask)
    # if d==1:
    #     cv2.imshow('white',imgresult)
    # if d==2:
    #     cv2.imshow('yellow',imgresult)
    if d==3:
        cv2.imshow('green',imgresult)
    # if d==4:
    #     cv2.imshow('red',imgresult)
    # if d==5:
    #     cv2.imshow('pink',imgresult)
    imggray = cv2.cvtColor(imgresult, cv2.COLOR_BGR2GRAY)
    imgblur = cv2.GaussianBlur(imggray, (7, 7), 1)
    imgcanny = cv2.Canny(imgblur, 50, 50)
    count=0
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        # if d==5:
        #     cv2.imshow('pinkk',imgresult)
        # if d==4:
        #     print(area,count)
        #     count=count+1
        if (area > 100):
            # cv2.drawContours(imgcontours, cnt, -1, (255, 0, 0), 3)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            objapp = len(approx)
            M = cv2.moments(cnt)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            y = int((cx - 10) / h1)
            x = int((cy - 10) / w1)
            # if objapp == 4:
            res[x][y][2] = d
            if(d==3):
                print(area,x,y,count)
                count=count+1
            res[x][y][0] = cx
            res[x][y][1] = cy
            if (d == 5):
                res[x][y][3] = -1
            # print([x,y])
    return (res)


def onewaybluepath(res, imgtest, lower, upper):
    imghsv = cv2.cvtColor(imgtest, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(imghsv, lower, upper)
    imgresult = cv2.bitwise_and(imgtest, imgtest, mask=mask)
    cv2.imshow('blue', imgresult)
    imggray = cv2.cvtColor(imgresult, cv2.COLOR_BGR2GRAY)
    imgblur = cv2.GaussianBlur(imggray, (7, 7), 1)
    imgcanny = cv2.Canny(imgblur, 50, 50)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # print("me blue ka area hoon")
        # print(area)
        if (area > 100):
            d = 0
            # cv2.drawContours(imgcontours, cnt, -1, (255, 0, 0), 3)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
            objapp = len(approx)
            M = cv2.moments(cnt)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            y = int((cx - 10) / h1)
            x = int((cy - 10) / w1)
            if objapp == 3:
                # print("asa")
                temp = 0
                count = 0
                for i in range(0, 3, 1):
                    # print(approx[i][0][0])
                    # print(cX)
                    if (approx[i][0][0] > cx - 6 and approx[i][0][0] < cx + 6):
                        temp = i
                        count = 1
                # print("temp")
                if (count == 1):
                    # print(approx[temp][0][1])
                    if (approx[temp][0][1] > cy):
                        d = 504  # south
                    elif (approx[temp][0][1] < cy):
                        d = 503  # north
                else:
                    for i in range(0, 3, 1):
                        # print(approx[i][0][1])
                        # print(cX)
                        if (approx[i][0][1] > cy - 6 and approx[i][0][1] < cy + 6):
                            temp = i
                            count = 1
                    # print("temp")
                    if (count == 1):
                        # print(approx[temp][0][0])
                        if (approx[temp][0][0] > cy):
                            d = 501  # east
                        elif (approx[temp][0][0] < cx):
                            d = 502  # west

                res[x][y][3] = d
                # if(d==0):
                # print("notposs")
            elif objapp == 4:
                res[x][y][3]=-1
                res[x][y][2] = 600  # blue square
            else:
                res[x][y][3] = -1
                res[x][y][2] = 700  # blue circle
            res[x][y][0] = cx
            res[x][y][1] = cy
            # print([x, y])
    return (res)


def initialmatrix(res, pic):
    imgtest = pic

    # now for red green white and yellow squares
    # 1. red
    lower1 = np.array([0, 70, 50])
    upper1 = np.array([10, 255, 255])
    res = forsquares(res, imgtest, lower1, upper1, 4)

    # 2. green
    lower = np.array([36, 40, 40], np.uint8)
    upper = np.array([70, 255, 255], np.uint8)
    res = forsquares(res, imgtest, lower, upper, 2)

    # 3. yellow
    lower = np.array([22, 93, 0], np.uint8)
    upper = np.array([45, 255, 255], np.uint8)
    res = forsquares(res, imgtest, lower, upper, 3)

    # 4. white
    lower = np.array([0, 0, 168])
    upper = np.array([172, 111, 255])
    res = forsquares(res, imgtest, lower, upper, 1)

    # 5. Pink
    lower = np.array([133, 25, 183], np.uint8)
    upper = np.array([166, 255, 255], np.uint8)
    res = forsquares(res, imgtest, lower, upper, 5)

    # 6 blue triangle for one way and diresction with them 501 502 503 504 for EWNS
    # also for hospitals as they are also blue

    lower = np.array([110, 50, 50])
    upper = np.array([130, 255, 255])
    res = onewaybluepath(res, imgtest, lower, upper)


def adjacencymaker(res, kores, n):
    k = n * n
    for i in range(0, n):
        for j in range(0, n):
            m = (i * n) + j
            dir1 = res[i][j][3]
            li = i
            lj = j - 1
            ri = i
            rj = j + 1
            ui = i - 1
            uj = j
            di = i + 1
            dj = j
            if (li >= 0 and li < n and lj >= 0 and lj < n):
                dir2 = res[li][lj][3]
                m1 = (li * n) + lj
                val = 0
                if (dir2 != 501 and (dir1 == 0 or dir1 == 502 or dir1==-1) and dir2!=-1):
                    val = res[i][j][2]
                if (val == 0):
                    val = 0
                if (res[li][lj][2] == 0):
                    val = 0
                kores[m][m1] = val
            if (ri >= 0 and ri < n and rj >= 0 and rj < n):
                dir2 = res[ri][rj][3]
                m1 = (ri * n) + rj
                val = 0
                if (dir2 != 502 and (dir1 == 0 or dir1 == 501 or dir1==-1) and dir2!=-1):
                    val = res[i][j][2]
                if (val == 0):
                    val = 0
                if (res[ri][rj][2] == 0):
                    val = 0
                kores[m][m1] = val
            if (ui >= 0 and ui < n and uj >= 0 and uj < n):
                dir2 = res[ui][uj][3]
                m1 = (ui * n) + uj
                val = 0
                if (dir2 != 504 and (dir1 == 0 or dir1 == 503 or dir1==-1) and dir2!=-1):
                    val = res[i][j][2]
                if (val == 0):
                    val = 0
                if (res[ui][uj][2] == 0):
                    val = 0
                kores[m][m1] = val
            if (di >= 0 and di < n and dj >= 0 and dj < n):
                dir2 = res[di][dj][3]
                m1 = (di * n) + dj
                val = 0
                if (dir2 != 503 and (dir1 == 0 or dir1 == 504 or dir1==-1) and dir2!=-1):
                    val = res[i][j][2]
                if (val == 0):
                    val = 0
                if (res[di][dj][2] == 0):
                    val = 0
                kores[m][m1] = val


def adj_tuple(kores, n):
    lst = []
    for i in range(n * n):
        for j in range(n * n):
            if kores[i][j] > 0:
                lst.append((i, j, kores[i][j]))

    return lst


from collections import deque, namedtuple

# we'll use infinity as a default distance to nodes.
inf = float('inf')
Edge = namedtuple('Edge', 'start, end, cost')


def make_edge(start, end, cost=1):
    return Edge(start, end, cost)


class Graph:
    def __init__(self, edges):
        # let's check that the data is right
        wrong_edges = [i for i in edges if len(i) not in [2, 3]]
        if wrong_edges:
            raise ValueError('Wrong edges data: {}'.format(wrong_edges))

        self.edges = [make_edge(*edge) for edge in edges]

    @property
    def vertices(self):
        return set(
            sum(
                ([edge.start, edge.end] for edge in self.edges), []
            )
        )

    def get_node_pairs(self, n1, n2, both_ends=True):
        if both_ends:
            node_pairs = [[n1, n2], [n2, n1]]
        else:
            node_pairs = [[n1, n2]]
        return node_pairs

    def remove_edge(self, n1, n2, both_ends=True):
        node_pairs = self.get_node_pairs(n1, n2, both_ends)
        edges = self.edges[:]
        for edge in edges:
            if [edge.start, edge.end] in node_pairs:
                self.edges.remove(edge)

    def add_edge(self, n1, n2, cost=1, both_ends=True):
        node_pairs = self.get_node_pairs(n1, n2, both_ends)
        for edge in self.edges:
            if [edge.start, edge.end] in node_pairs:
                return ValueError('Edge {} {} already exists'.format(n1, n2))

        self.edges.append(Edge(start=n1, end=n2, cost=cost))
        if both_ends:
            self.edges.append(Edge(start=n2, end=n1, cost=cost))

    @property
    def neighbours(self):
        neighbours = {vertex: set() for vertex in self.vertices}
        for edge in self.edges:
            neighbours[edge.start].add((edge.end, edge.cost))

        return neighbours

    def dijkstra(self, source, dest):
        # print(source,self.vertices)
        assert source in self.vertices, 'Such source node doesn\'t exist'
        distances = {vertex: inf for vertex in self.vertices}
        previous_vertices = {
            vertex: None for vertex in self.vertices
        }
        distances[source] = 0
        vertices = self.vertices.copy()

        while vertices:
            current_vertex = min(
                vertices, key=lambda vertex: distances[vertex])
            vertices.remove(current_vertex)
            if distances[current_vertex] == inf:
                break
            for neighbour, cost in self.neighbours[current_vertex]:
                alternative_route = distances[current_vertex] + cost
                if alternative_route < distances[neighbour]:
                    distances[neighbour] = alternative_route
                    previous_vertices[neighbour] = current_vertex

        path, current_vertex = deque(), dest
        while previous_vertices[current_vertex] is not None:
            path.appendleft(current_vertex)
            current_vertex = previous_vertices[current_vertex]
        if path:
            path.appendleft(current_vertex)
        return path


def pinklist(res):
    lst = []
    all_pink = []
    for i in range(0, n, 1):
        for j in range(0, n, 1):
            if (res[i][j][2] == 5):
                # res[i][j][2]=-5 # this pink is of no use once we store its neighbours in a list
                l = i + 1
                m = j
                if (l < n and res[l][m][2] != 0 and res[l][m][2] < 5):
                    lst.append([l, m])
                l = i - 1
                m = j
                if (l >= 0 and res[l][m][2] != 0 and res[l][m][2] < 5):
                    lst.append([l, m])
                l = i
                m = j + 1
                if (m < n and res[l][m][2] != 0 and res[l][m][2] < 5):
                    lst.append([l, m])
                l = i
                m = j - 1
                if (m >= 0 and res[l][m][2] != 0 and res[l][m][2] < 5):
                    lst.append([l, m])
                all_pink.append([i, j])
    return lst, all_pink


def hospos(val, res):
    if val == 11:
        pal = 600  # square
    else:
        pal = 700  # circle
    lst = []
    all_hos = []
    for i in range(0, n, 1):
        for j in range(0, n, 1):
            if (res[i][j][2] == pal):
                # res[i][j][2]=-5 # this pink is of no use once we store its neighbours in a list
                l = i + 1
                m = j
                if (l < n and res[l][m][2] != 0 and res[l][m][2] < 5):
                    lst.append([l, m])
                l = i - 1
                m = j
                if (l >= 0 and res[l][m][2] != 0 and res[l][m][2] < 5):
                    lst.append([l, m])
                l = i
                m = j + 1
                if (m < n and res[l][m][2] != 0 and res[l][m][2] < 5):
                    lst.append([l, m])
                l = i
                m = j - 1
                if (m >= 0 and res[l][m][2] != 0 and res[l][m][2] < 5):
                    lst.append([l, m])
                all_hos.append([i, j])
    return lst, all_hos


def search(list, platform):  # python function to search for an element in a list
    for i in range(len(list)):
        if list[i] == platform:
            return True
    return False


def finalhospi(img):
    imghsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    lower = np.array([110, 50, 50])
    upper = np.array([130, 255, 255])
    mask=cv2.inRange(imghsv,lower,upper)
    countours,_= cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in countours:
        area=cv2.contourArea(cnt)
        if(area>100):
            peri= cv2.arcLength(cnt, True)
            approx= cv2.approxPolyDP(cnt,0.04*peri, True)
            objapp= len(approx)
            if(objapp==4):
                return(11) #square
            else:
                return (12) #circle

def oneround(res, graph, kores, curr, count):  # function for one round i.e getting to the patient then takiing him to hospi

    # currpos= it is the postion of the bot on the start of the motion or the new cycle currpos is in the int form
    currpos = curr
    # currpos=35
    # step 1
    pink, all_pink = pinklist(res)  # step 1 first of all we search for the pink color square and store its neighbour in a list
    # print(pink)

    paths = []
    dist = []
    # print(item)
    for item in pink:  # lets select the best nearest vertics
        pth = graph.dijkstra(currpos, item[0] * n + item[1])
        paths.append(pth)
        t = 0
        for i in range(1, len(pth), 1):
            t += kores[pth[i - 1]][pth[i]]
        dist.append(t)
    print("topink")
    print(dist)
    min_pos = dist.index(min(dist))

    runonme = paths[min_pos]  # this is the path which the bot follows
    x1 = pink[min_pos][0]  # coordinates of the point where the bot will stop for the pink rectangle
    y1 = pink[min_pos][1]

    print(runonme)
    letsgohusky(runonme, 12)
    # step 2 find the patient position i.e adjacent PINK block
    p1 = 0  # [p1,p2] coordinates of the pink position
    p2 = 0
    if search(all_pink, [x1 + 1, y1]):
        res[x1 + 1][y1][2] = -5
        p1 = x1 + 1
        p2 = y1
    elif search(all_pink, [x1 - 1, y1]):
        res[x1 - 1][y1][2] = -5
        p1 = x1 - 1
        p2 = y1
    elif search(all_pink, [x1, y1 + 1]):
        res[x1][y1 + 1][2] = -5
        p1 = x1
        p2 = y1 + 1
    elif search(all_pink, [x1, y1 - 1]):
        res[x1][y1 - 1][2] = -5
        p1 = x1
        p2 = y1 - 1

    # step 3 reveal the patient condition and store its type in the i.e square for covid circle for non covid
    # val = 11  # blue square
    # val= 12 #blue circle
    x = 0
    while True:
        p.stepSimulation()
        if x == 5000:
            env.remove_cover_plate(p1,p2)
            break
        x += 1
    time.sleep(1)
    img2=env.camera_feed()
    img3=img2[45+(p1*53)-30:45+(p1*53)+30, 45+(p2*53)-30:45+(p2*53)+30]

    val=finalhospi(img3)
    print("hospino->",val)
    # step 3.1 move the bot from [x1][y1] to [p1][p2]
    print([p1, p2])
    # p1 = x1  # for testing
    # p2 = y1  # for testing
    letsgohusky([x1 * 12 + y1, p1 * 12 + p2], 12)
    time.sleep(1)
    # step 4 now the bot will go from the position [p1][p2] to the postion adjacent to the hospital
    hospital_list, all_hos = hospos(val, res)
    paths.clear()
    dist.clear()
    for item in hospital_list:  # lets select the best nearest position of the hospital
        pth = graph.dijkstra(p1 * n + p2, item[0] * n + item[1])
        paths.append(pth)
        t = 0
        for i in range(1, len(pth), 1):
            t += kores[pth[i - 1]][pth[i]]
        dist.append(t)
    print("tohospi")
    print(dist)
    min_pos = dist.index(min(dist))
    runonme = paths[min_pos]  # this is the path which the bot follows to go from patient to respective hospital
    x1 = hospital_list[min_pos][0]  # coordinates of the point where the bot will stop for the hospital
    y1 = hospital_list[min_pos][1]

    # step 4.1 bot will run on the path runonme
    print(runonme)
    letsgohusky(runonme,12)

    # step 5 move the bot on the hospital
    p1 = 0  # [p1,p2] coordinates of the position of hospital
    p2 = 0
    if search(all_hos, [x1 + 1, y1]):
        # res[x1 + 1][y1][2] = -800
        p1 = x1 + 1
        p2 = y1
    elif search(all_hos, [x1 - 1, y1]):
        # res[x1 - 1][y1][2] = -800
        p1 = x1 - 1
        p2 = y1
    elif search(all_hos, [x1, y1 + 1]):
        # res[x1][y1 + 1][2] = -800
        p1 = x1
        p2 = y1 + 1
    elif search(all_hos, [x1, y1 - 1]):
        # res[x1][y1 - 1][2] = -800
        p1 = x1
        p2 = y1 - 1

    # step 5.2 move the bot from [x1][y1] to [p1][p2]
    print("call")
    print([p1, p2])
    letsgohusky([x1*12+y1,p1*12+p2],12)
    curr= p1*12+p2
    # p1=curr
    if count==0:
        oneround(res, graph, kores, curr,1)


def arena_make(env):
    frame = env.camera_feed(is_flat=True)
    fromCenter = False
    showCrosshair = False
    r = cv2.selectROI('image', frame, fromCenter, showCrosshair)
    img = frame[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
    return r, img


if __name__ == "__main__":

    parent_path = os.path.dirname(os.getcwd())
    os.chdir(parent_path)
    env = gym.make("pix_main_arena-v0")
    r, arena = arena_make(env)
    # img = cv2.cvtColor(arena, cv2.COLOR_BGR2HSV)
    # cv2.imshow('img',img)



    # print(pic.size)
    # pic=cv2.imread('12x12v.png')
    pic=arena
    # pic = cv2.resize(pic, (700, 700))
    np.set_printoptions(threshold=sys.maxsize)
    n = 12
    res = np.zeros((n, n, 4), dtype=int)
    h1 = pic.shape[0] / n
    w1 = pic.shape[1] / n
    h1 -= 2
    w1 -= 2
    # print(res)
    # res2=res
    initialmatrix(res, pic)
    print(h1)
    print(w1)

    cv2.imshow('char', pic)
    # pic2 = cv2.imread('6x6v.png')
    # cv2.imshow('char2', pic2)

    # kores=adjacency matrix

    res[7][4][3] = 501  # hardcodefornow
    res[6][0][3] = 504  # hardcodefornow
    res[5][9][3] = 503  # hardcodefornow
    res[1][6][3] = 502  # hardcodefornow
    # res[5][11][2]=-5    # hardcodefornow

    kores = np.zeros((n * n, n * n), dtype=int)
    adjacencymaker(res, kores, n)
    # print(res[1][3][0])
    # print(kores)
    # print("now")
    # print([kores[33][32],kores[12][15],kores[13][12],kores[18][19],kores[19][18],kores[19][13]])

    newlist = adj_tuple(kores, n)
    graph = Graph(newlist)
    # print(graph.dijkstra(35,0))
    # print(newlist)
    # res[7][4][3]=501
    # print(res)
    time.sleep(20)
    oneround(res, graph, kores,curr,0)
    # oneround(res, graph, kores,curr)
    # print(res[6][0][3])

    # print(res)
    if (cv2.waitKey(0) & 0xFF == 27):
        cv2.destroyAllWindows()
