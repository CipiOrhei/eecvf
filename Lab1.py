from skimage import io 
import matplotlib.pylab as plt
import numpy as np


def load_image_from_file(path):
    """
    load the image from a file located at path
    """
    
    img = io.imread(path)
    #find information about image dimension
    print(img.shape)
    '''
        https://en.wikipedia.org/wiki/Portable_Network_Graphics
        Number of channels PNG
        1 grayscale
        2 grayscale and alpha: level of opacity
        3 truecolor: red, green and blue: rgb
        4 red, green, blue and alpha
    '''
    
    return img

def save_pixels_to_image(matrix, jpg_name):
    nparr = np.array(matrix, dtype=np.uint8)
    bmp = io.imsave(jpg_name, nparr)

def only_truecolors(pngfile):
    img = load_image_from_file(pngfile)
    lns = img.shape[0]
    cls = img.shape[1]
    a = [[0]*cls for i in range(lns)]
    for i in range(lns):
        for j in range(cls):
            # print(img[i][j])
            [r, g, b, alpha] = img[i][j]
            a[i][j] = [r, g, b]
    return a


def inverse_image(img):
    """
    transform image
    """
    lns = img.shape[0]
    cls = img.shape[1]
    a = [[0]*cls for i in range(lns)]
    for i in range(lns):
        for j in range(cls):
            #print(img[i][j])
            [r, g, b] = img[i][j]
            a[i][j] = [255 - r,255 - g, 255 - b]
    return a

def transform_image(img, r,g,b):
    """
    add a constant integer to each pixel color
    """
    lns = img.shape[0]
    cls = img.shape[1]
    a = [[0]*cls for i in range(lns)]
    #print(a)
    for i in range(lns):
        for j in range(cls):
            #a[i][j] = [img[i][j][0],0,0]
            
            [rr, gg, bb] = img[i][j]
            
            a[i][j] = [(rr + r) % 256, (gg + g) % 256, (bb + b) % 256]
    
    #print(a)
    return a


def filter_color(img, color='red'):
    '''
    split the image by color planes
    '''
    lns = img.shape[0]
    cls = img.shape[1]
    a = [[0]*cls for i in range(lns)]
    for i in range(lns):
        for j in range(cls):
            r = img[i][j][0]
            g = img[i][j][1]
            b = img[i][j][2]
         
            if color == 'red':
                a[i][j] = [r, r, r]
                    
            if color == 'green':
                a[i][j] = [g, g, g]
            
            if color == 'blue':
                a[i][j] = [b, b, b]

    return a

def filter_gray(img):
    '''
    show the grayscale image
    '''
    lns = img.shape[0]
    cls = img.shape[1]
    a = [[0]*cls for i in range(lns)]
    for i in range(lns):
        for j in range(cls):
            [r, g, b] = img[i][j]
            avg = (int(r) + g + b)//3
            a[i][j] = [avg, avg, avg]
    return a

def plane_image(img):
    """
    show the bitplanes
    """
    linii = img.shape[0]
    coloane = img.shape[1]
    a=[[[0]*coloane for i in range (linii)] for j in range(8)]
    for i in range(linii):
        for j in range(coloane):
            for b in range(8):
                bit = ((img[i][j][0] >> b ) & 1)  * 255
                a[b][i][j] = [bit,  bit, bit]
    return a

def displayHistogram(bins, values):
    '''
    display the histogram
    '''
    plt.bar(bins, values, 1, color="black")
    
def histogram(im):
    '''
    compute the histograme of an image
    '''
    h = [0] * 256
    n = len(im)
    m = len(im[0])
    for i in range(n):
        for j in range(m):
            h[im[i][j][0]]+=1
    bins = list(range(0,256))
    displayHistogram(bins, h)
    

def main():
    #1. Load an image
    plt.subplot(1,3,1)
    car = load_image_from_file("TestData/labs/lab1/SKI27.png")

    #2. Save only TrueColors
    rbg = only_truecolors("TestData/labs/lab1/SKI27.png")
    save_pixels_to_image(rbg, "TestData/labs/lab1/SKI27.jpg")
    plt.imshow(rbg)
    

    #3. Diplay the image on truecolors RGB
    car = load_image_from_file("TestData/labs/lab1/SKI27.jpg")
    plt.imshow(car)
    black_car = inverse_image(car)
    save_pixels_to_image(black_car, "TestData/labs/lab1/SKI27_inverse.jpg")
    #display the image
    plt.subplot(1,3,2)
    plt.imshow(black_car)

    #4. Display the invers and original image
    #imshow

    #5. How can we create a different car? another color
    car = load_image_from_file("TestData/labs/lab1/SKI27.jpg")
    another_car = transform_image(car, r = 10, g = 200, b = 45)
    save_pixels_to_image(another_car, "TestData/labs/lab1/SKI27_another.jpg")
    

    #6. Make 20 different cars!

    #7. What is wrong with you? Let the pixels not changed 
    #car = load_image_from_file("TestData/labs/lab1/SKI27.jpg")
    another_car = transform_image(car, r = 50, g = 20, b = 245)
    save_pixels_to_image(another_car, "TestData/labs/lab1/SKI27_another2.jpg")

    #8. How can we see the RGB colors?
    #car = load_image_from_file("TestData/labs/lab1/SKI27.jpg")
    color = "red"
    color_plane_car = filter_color(car, color)
    save_pixels_to_image(color_plane_car, "TestData/labs/lab1/SKI27_" + color + ".jpg")

    color = "green"
    color_plane_car = filter_color(car, color)
    save_pixels_to_image(color_plane_car, "TestData/labs/lab1/SKI27_" + color + ".jpg")

    color = "blue"
    color_plane_car = filter_color(car, color)
    save_pixels_to_image(color_plane_car, "TestData/labs/lab1/SKI27_" + color + ".jpg")

    #9. What is the greay scale image?
    #car = load_image_from_file("TestData/labs/lab1/SKI27.jpg")
    grayscale_car = filter_gray(car)
    save_pixels_to_image(grayscale_car, "TestData/labs/lab1/SKI27_grayscale.jpg")

    #10. What is and bitplane? How are they look like?
    bit_planes = plane_image(car)
    for index in range(len(bit_planes)):
        save_pixels_to_image(bit_planes[index], "TestData/labs/lab1/SKI27_bit_plane_"+ str(index) +".jpg")

    #11. We need the image histogram? What's that?!
    histogram(car)

if __name__ == "__main__":
    main()
