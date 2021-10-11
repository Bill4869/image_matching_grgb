import cv2
import matplotlib.pyplot as plt
# import numpy as np

def orb_match(img1, img2, real1, real2):
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)

    good = []
    try:
        matches = bf.knnMatch(des1, des2, k=2)
        for m,n in matches:
            if m.distance < 0.8*n.distance:
                good.append([m])
    except:
        pass
    img = cv2.drawMatchesKnn(real1,kp1,real2,kp2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    print('kp1 : {}'.format(len(kp1)))
    print('kp2 : {}'.format(len(kp2)))
    print('match points : {}'.format(len(good)))

    rate = 0
    if(len(kp1) + len(kp2) != 0):
        rate = len(good) * 2 / (len(kp1) + len(kp2)) * 100
    print("match : {}".format(rate))

    return img, rate

imgs = []
path = './images/sketch'
for i in range(1, 4):
    img = cv2.imread(path + '{}.jpg'.format(i), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
    imgs.append(img)

orb = cv2.ORB_create()

# brute-force matcher
bf = cv2.BFMatcher()
# bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
# bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

count = 1
fig, axes = plt.subplots(2, 2)
for i in range(len(imgs)):
    for j in range(i, len(imgs)):
        if(i == j):
            continue
        
        print('{} : {}'.format(i, j))
        img1 = imgs[i]
        img2 = imgs[j]

        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        b1 = img1.copy()
        b2 = img2.copy()
        b1[:, :, 1] = 0
        b1[:, :, 2] = 0
        b2[:, :, 1] = 0
        b2[:, :, 2] = 0

        g1 = img1.copy()
        g2 = img2.copy()
        g1[:, :, 0] = 0
        g1[:, :, 2] = 0
        g2[:, :, 0] = 0
        g2[:, :, 2] = 0

        r1 = img1.copy()
        r2 = img2.copy()
        r1[:, :, 0] = 0
        r1[:, :, 1] = 0
        r2[:, :, 0] = 0
        r2[:, :, 1] = 0

        gray_match, gray_rate = orb_match(gray1, gray2, img1, img2)
        gray_match = cv2.cvtColor(gray_match, cv2.COLOR_BGR2RGB)

        b_match, b_rate = orb_match(b1, b2, img1, img2)
        b_match = cv2.cvtColor(b_match, cv2.COLOR_BGR2RGB)

        g_match, g_rate = orb_match(g1, g2, img1, img2)
        g_match = cv2.cvtColor(g_match, cv2.COLOR_BGR2RGB)

        r_match, r_rate = orb_match(r1, r2, img1, img2)
        r_match = cv2.cvtColor(r_match, cv2.COLOR_BGR2RGB)

        axes[0, 0].imshow(gray_match)
        axes[0, 0].set_title('grayscale, match(%): {:.3f}'.format(gray_rate))
        axes[0, 1].imshow(b_match)
        axes[0, 1].set_title('blue channel, match(%): {:.3f}'.format(b_rate))
        axes[1, 0].imshow(g_match)
        axes[1, 0].set_title('green channel, match(%): {:.3f}'.format(g_rate))
        axes[1, 1].imshow(r_match)
        axes[1, 1].set_title('red channel, match(%): {:.3f}'.format(r_rate))

        plt.savefig('./output/{}.png'.format(str(count)))
        count += 1



