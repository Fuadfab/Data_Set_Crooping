import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
import glob

final_word = []

def word_segmentation_two(image,size,letter,start,end):
    global final_word
    h,w = image.shape
    img_array =np.zeros((size,w), dtype="uint8")
    x = 0
    y = 0
    for i in range(h):
        if(i>=start and i<end):
            y = 0
            for j in range(w):
                img_array[x][y] = image[i][j]
                y = y + 1
            
            x = x + 1

    h,w = img_array.shape
    img = np.stack((img_array,) * 3,-1)
    img = img.astype(np.uint8)
    grayed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grayed = cv2.resize(grayed, dsize =(w,100), interpolation = cv2.INTER_AREA)
    #plt.imshow(grayed)
    #plt.show()
    final_word.append(grayed)

def word_segmentation_one(image,size,start,end):
    h,w = image.shape
    img_array = np.zeros((h,size), dtype="uint8")
    x = 0
    y = 0
  
    for i in range(h):
        y = 0
        for j in range(w):
            if(j>=start and j<end):
                img_array[x][y] = image[i][j]
                y = y + 1
            
        x = x + 1
    h2,w2 = img_array.shape

    img_array_rry = np.array(img_array)

    for i in range(h2):
        for j in range(w2):
            if(img_array[i][j]>=15):
                img_array[i][j] = 255
            else:
                img_array[i][j] = 0   


    cout = 0
    c = 0
    gap = []

    gap.append(0)
    for i in range(h2):
        cout = 0
        for j in range(w2):
            if(img_array[i][j] == 0):
                cout = cout +1
                if cout >= w:
                    c = c + 1
                    gap.append(i)

    gap.append(h2)
    
    word_size = []
    word_start =[]
    word_end = []
    total_gap = 0
    for i in range(len(gap)-1):
        if(gap[i+1] - gap[i]== 1):
            total_gap = total_gap + 1
        if(gap[i+1] - gap[i]>2):
            word_size.append(gap[i+1] - gap[i])
            word_start.append(gap[i])
            word_end.append(gap[i+1])
   
    for i in range(len(word_size)):
        word_segmentation_two(img_array_rry,word_size[i],size,word_start[i],word_end[i])


def word_segmentation(image):
    word_gap = []
    c = 0
    cout = 0
    sum = []
    null_list = []
    h,w = image.shape
    copy = image
    sum.append(copy.sum(axis = 0))
    a = np.asarray(sum)
    m,n = a.shape
    #print(m,n)
    gapp = []
    for i in range(m):
        for j in range(n-1):
            if(a[i][j]<5):
                null_list.append(a[i][j])
                gapp.append(j)
    
    gapp.append(w)
    word_size = []
    word_start =[]
    word_end = []
    for i in range(len(gapp)-1):
        if(gapp[i+1] - gapp[i]>5):
            word_size.append(gapp[i+1] - gapp[i])
            word_start.append(gapp[i])
            word_end.append(gapp[i+1])
    
    for i in range(len(word_size)):
        word_segmentation_one(image,word_size[i],word_start[i],word_end[i])

def line_segmentation(image,size,letter,start,end):
    h,w = image.shape
    img_array =np.zeros((size,w), dtype="uint8")
    x = 0
    y = 0

    for i in range(h):
        if(i>=start and i<end):
            y = 0
            for j in range(w):
                img_array[x][y] = image[i][j]
                y = y + 1 
            x = x + 1
    
    return img_array


from PIL import Image

def function(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h,w = gray.shape

    img_array = np.zeros((h,w), dtype="uint8")

    for i in range(h):
        for j in range(w):
            if gray[i][j]>25:
                img_array[i][j] = gray[i][j]
            else:
                img_array[i][j] = 0

    for i in range(h):
        for j in range(w):
            if img_array[i][j]==0:
                img_array[i][j] = 255
            else:
                img_array[i][j] = 0 

    

    image = cv2.resize(img_array, dsize =(1000, 150), interpolation = cv2.INTER_AREA)

    h,w = image.shape
    cout = 0
    c = 0
    gap = []


    gap.append(0)
    for i in range(h):
        cout = 0
        for j in range(w):
            if(image[i][j] == 0):
                cout = cout + 1
                if cout >= w-20:
                    c = c + 1
                    gap.append(i)


    gap.append(h)
    letter_size = []
    line_start =[]
    line_end = []
    for i in range(len(gap)-1):
        if(gap[i+1] - gap[i]>5):
            letter_size.append(gap[i+1] - gap[i])
            line_start.append(gap[i])
            line_end.append(gap[i+1])

        
    crop_image = []
    for i in range(len(letter_size)):
        crop = line_segmentation(image,letter_size[i],letter_size,line_start[i],line_end[i])
        h,w = crop.shape
        grayed = cv2.resize(crop, dsize =(1000, 150), interpolation = cv2.INTER_AREA)
        crop_image.append(grayed)

    for i in range(len(crop_image)):
        word_segmentation(crop_image[i])


images = [cv2.imread(file) for file in glob.glob("I://Data For Math//Data//*.png")]
print(len(images))

for i in range(len(images)):
    function(images[i])

outpath ="I:\\Data For Math\\Clean Data\\"
idx = 0	 
model_image = []
img_width = []
for i in range(len(final_word)):
    img = final_word[i]
    cv2.imwrite(outpath + str(idx) + '.jpg', img)
    idx = idx + 1
