import random
import torch


def randomShape1(l,b,percentage):
    weight = torch.ones(l,b).cuda()
    n = int(l*b*percentage)
    randomlist = random.sample(range(0,l*b),n)
    for i in randomlist:
        x = i//l
        y = i%b  
        weight[x][y] = 0
    return weight 

def randomShape(a,c,l,b,percentage):
    # mention percentage btw 0-1 to number of zeros
    weight = torch.ones(a,c,l,b).cuda()
    n = int(a*c*l*b*percentage)
    randomlist = random.sample(range(0,a*c*l*b),n)
    for i in randomlist:
        x1 = i//(b*c*l)
        x2 = (i%(c*l*b))//(l*b)
        x3 = (i%(c*l*b))%(l*b)//l
        x4 = (i%(c*l*b))%(l*b)%b
        #x = i//l
        #y = i%b
        weight[x1][x2][x3][x4] = 0
    return weight

def swastik(l):
    # all ones in 3*3
    b = l
    weight = torch.zeros(l,b).cuda()
    for i in range(0,l):
        weight[l//2][i] = 1
        if(i<=l//2):
            weight[i][0] = 1
            weight[l-i-1][b-1] = 1
    for i in range(0,b):
        weight[i][b//2] = 1
        if(i>=b//2):
            weight[0][i] = 1
            weight[l-1][b-i-1] = 1
    if(l==3):
        weight[0][1] = weight[2][1] = weight[1][0] = weight[1][2] = 0
    return weight

def star(l):
    #assumed it's always even shaped
    # all ones in 3*3
    weight = torch.zeros(l,l).cuda()
    for i in range(0,l):
        weight[i][l//2] = 1
        weight[l//2][i] = 1
        weight[i][i] = 1
        weight[l-i-1][i] = 1
    if(l==3):
        weight[0][1] = weight[2][1] = weight[1][0] = weight[1][2] = 0
    return weight

def circle(r):
    return oval(r,r)

def oval(l,b):
    weight = torch.ones(l,b).cuda()
    for i in range(0,l//2):
        if(b%2):
            for j in range(0,b//2-i):
                weight[i][j] = 0
                weight[i][b-1-j] = 0
        else:
            for j in range(0,b//2-1-i):
                weight[i][j] = 0
                weight[i][b-1-j] = 0
                
    for i in range(l-1,l//2,-1):
        if(b%2):
            for j in range(0,b//2+i-l+1):
                weight[i][j] = 0
                weight[i][b-1-j] = 0
        else:
            for j in range(0,b//2+i-l):
                weight[i][j] = 0
                weight[i][b-1-j] = 0
    return weight
def Ishape(r):
    weight = torch.zeros(r,r).cuda()
    for i in range(0,r):
        weight[0][i] = 1
        weight[r-1][i] = 1
        weight[i][r//2] = 1
    return weight

def twocircleshape(r):
    weight = torch.zeros(r,r).cuda()
    for i in range(0,r//2):
        for j in range(0,r//2+1):
            weight[i][j] = 1
            weight[r-i-1][r-j-1] = 1
    weight[r//2][r//2] = 1
    return weight

def stackTimes(a,x):
    b = a
    a = torch.cat([a[None,:,:],a[None,:,:]],dim=0)
    for i in range(1,x-1):
        a = torch.cat([a[:,:,:],b[None,:,:]],dim=0)
    return a

def oneShape():
    weight = torch.zeros(5,5).cuda()
    for i in range(0,5):
        weight[i][2] = 1 
    return weight
 
def twoShape():
    weight = torch.zeros(5,5).cuda()
    for i in range(0,5):
        weight[0][i] = 1
        weight[4][i] = 1
        weight[i][4-i] = 1
    return weight

def threeShape():
    weight = torch.zeros(5,5).cuda()
    for i in range(0,5):
        weight[0][i] = 1
        weight[4][i] = 1
        weight[i][4] = 1
    weight[2][2] = weight[2][3] = 1
    return weight

def fourShape():
    weight = torch.zeros(5,5).cuda()
    for i in range(0,5):
        weight[i][0] = 1
        weight[3][i] = 1
    weight[2][3] = weight[4][3] = 1
    return weight

def fiveShape():
    weight = torch.zeros(5,5).cuda()
    for i in range(0,5):
        weight[0][i] = 1
        weight[4][i] = 1
        weight[i][i] = 1
    return weight

def sixShape():
    weight = torch.zeros(5,5).cuda()
    for i in range(0,3):
        for j in range(0,3):
            weight[4-i][4-j] = 1
    weight[1][2] = weight[0][2] = 1
    return weight

def sevenShape():
    weight = torch.zeros(5,5).cuda()
    weight[0][0] = weight[0][1] = weight[0][2] = 1
    weight[0][2] = weight[1][2] = weight[2][2] = weight[3][2] = weight[4][2] = 1
    weight[4][1] = weight[4][3] = 1
    return weight

def eightShape():
    weight = torch.zeros(5,5).cuda()
    weight[0][2] = weight[2][2] = weight[4][2] = 1
    weight[1][1] = weight[1][3] = weight[3][1] = weight[3][3] = 1
    return weight

def nineShape():
    weight = torch.zeros(5,5).cuda()
    for i in range(0,3):
        for j in range(0,3):
            weight[i][j] = 1
            #weight[r-i-1][r-j-1] = 1
    weight[3][2] = weight[4][2] = 1
    return weight
    b = a
    a = torch.cat([a[None,:,:],a[None,:,:]],dim=0)
    for i in range(1,x-1):
        a = torch.cat([a[:,:,:],b[None,:,:]],dim=0)
    return a


def firstLayerMasking(r,num):
    circlePart = stackTimes(circle(r),num//2)
    iPart = stackTimes(Ishape(r),num//2)
    return torch.cat([circlePart,iPart],dim=0)[:,None,:,:]


def secondLayerMasking(r,num):
    circlePart = stackTimes(circle(r),num//4)
    twoCirclePart = stackTimes(twocircleshape(r),num//4)
    sevenPart = stackTimes(sevenShape(),num//4)
    fivePart = stackTimes(fiveShape(),num//4)
    return torch.cat([circlePart,twoCirclePart,sevenPart,fivePart],dim=0)[:,None,:,:]

def thirdLayerMasking(r,num):
    a1 = stackTimes(oneShape(),num//10)
    a2 = stackTimes(twoShape(),num//10)
    a3 = stackTimes(threeShape(),num//10)
    a4 = stackTimes(fourShape(),num//10)
    #a5 = stackTimes(fiveShape(),num//10)
    a6 = stackTimes(sixShape(),num//10)
    #a7 = stackTimes(sevenShape(),num//10)
    a8 = stackTimes(eightShape(),num//10)
    a9 = stackTimes(nineShape(),num//10)
    a0 = stackTimes(circle(r),num//10)
    return torch.cat([a1,a2,a3,a4,a6,a8,a9,a0],dim=0)[:,None,:,:]
