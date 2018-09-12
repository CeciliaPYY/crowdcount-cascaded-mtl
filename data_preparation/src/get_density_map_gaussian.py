import numpy as np

def fspecial(f_sz_x, f_sz_y , sig2):
    grd_x = (f_sz_x - 1.0)/2
    grd_y = (f_sz_y - 1.0)/2
    x,y = np.mgrid[-grd_x : grd_x+1:1 ,-grd_y:grd_y+1:1  ]
    dis = np.exp( (x**2 + y **2 )/  (-2 * sig2) )
    dis = dis / dis.sum()
    return dis

def get_density_map_gaussian(img,points):
    densemap = np.zeros(shape = img.shape[:2])
    if len(points)==0:
        return densemap

    if len(points)==1:
        for y,x in points:
            y = int(y)
            x = int(x)
            densemap[x,y] = 1
        return densemap
    else:
        f_sz = 15
        for y,x in points:
            x = int(x)
            y = int(y)
            band = int(f_sz / 2)
            h , w  = img.shape[:2]
            xmin , ymin =max( 0 , x - band ) , max( 0 ,  y - band)
            xmax , ymax = min( h -1 , x + band ), min( w - 1 ,  y+ band)
            sig2 = 4
            H = fspecial(xmax - xmin + 1 , ymax - ymin + 1 , sig2)
            densemap[xmin:(xmax+1),ymin:(ymax+1)] += H
        return densemap
