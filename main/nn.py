# -*- coding: utf-8 -*-
"""
Created on Tue Dec 02 00:47:40 2014

@author: Wasit
"""

from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import os
from scipy.spatial import kdtree
import cPickle
from scipy.ndimage import filters
import sys
import scipy.ndimage
sys.setrecursionlimit(10000)

rootdir="dataset"
#wd is level of the detail between 10 to 50
wd=30


def normFFT(im_file):
    im=np.array(Image.open(im_file).convert('L'))
    #converte image to frequency domain
    #f=np.log(np.abs(np.fft.fftshift(np.fft.fft2(im))))
    f=np.log(np.abs(np.fft.fft2(im)))
    #scaling
    s=(100./f.shape[0],100./f.shape[1])
    #normalized frequency domian
    return scipy.ndimage.zoom(f,s,order = 2)
    

    
def G(x,mu,s):
    return 1.0/ np.sqrt(2.0*np.pi)*np.exp(((x-mu)**2)/(-2.0*s**2))
def getHash3(im_file):
    f=normFFT(im_file)
    
    rmax,cmax=f.shape    
    sg=np.zeros((2*wd,wd))
    
    sg[0:wd,:]=np.log(np.abs(f[rmax-wd:rmax,0:wd]))
    sg[wd:2*wd,:]=np.log(np.abs(f[0:wd,0:wd]))
    filters.gaussian_filter(sg, (3,3), (0,0), sg)
    
#    ci=1
#    sg_hpf=np.concatenate((
#        np.reshape(sg[0:wd-ci,0:wd],(1,-1))[0],
#        np.reshape(sg[wd-ci:wd+ci,ci:wd],(1,-1))[0],
#        np.reshape(sg[wd+ci:2*wd,0:wd],(1,-1))[0]
#        ))
#    sg_hpf.astype(np.float32)
    
    #return sg_hpf/np.linalg.norm(sg_hpf)
    fsg=np.zeros(wd)
    for b in xrange(wd):
        for r in xrange(wd):
            for c in xrange(wd):
                rad=np.sqrt(r**2+c**2)            
                fsg[b]=fsg[b]+sg[wd+r,c]*G(rad,float(b),0.2)+sg[wd-r,c]*G(rad,float(b),0.2)
        fsg[b]=fsg[b]/(np.pi*float(b+1.0))
        fsg=fsg/np.linalg.norm(fsg)
        fsg.astype(np.float32)
    return fsg

def findall(tree,all_files):
    for f in all_files:
        fsg=getHash3(f)
        d,ids=tree.query(fsg,k=10)
        plt.figure(1)
        plt.clf()
        for i,index in enumerate(ids):
            print "%03d dis: %.3f %s"%(index,d[i],all_files[index])
            plt.subplot(2,5,i+1)
            plt.title("%03d dis: %.3f"%(index,d[i]),fontsize=10)
            im_icon=np.array(Image.open(all_files[index]).convert('L'))
            plt.imshow(im_icon)
            plt.axis('off')
        plt.set_cmap('gray')    
        plt.show()        
        plt.ginput(1)

if __name__ == '__main__':
    # read image to array
    
    all_files=[]
    for root, dirs, files in os.walk(rootdir):
        for f in files:
                if f.endswith('jpg') or f.endswith('JPG'):
                    all_files.append(os.path.join(root,f))
    #    for subdir in dirs:
    #        for iroot,idirs,ifiles in os.walk(os.path.join(root,subdir)):
    #            for f in ifiles:
    #                if f.endswith('jpg'):
    #                    all_files.append(os.path.join(iroot,f))
    # patch module-level attribute to enable pickle to work
    kdtree.node = kdtree.KDTree.node
    kdtree.leafnode = kdtree.KDTree.leafnode
    kdtree.innernode = kdtree.KDTree.innernode
    ####construct tree
    j=0;
    end=0
    while end is not 1:
        sub_bsg=[]
        sub_files=[]
        for i in xrange(2000):
            if len(all_files) is 0:
                end=1;            
                break
            f=all_files.pop(0)
            sub_files.append(f)    
            print '%02d %s'%(i,f)
            bsg=getHash3(f)    
            sub_bsg.append(bsg)
        tree = kdtree.KDTree(sub_bsg)
        pickleFile = open('%s/tree%02d.dat'%(rootdir,j), 'wb')
        cPickle.dump((sub_files,tree,wd), pickleFile, cPickle.HIGHEST_PROTOCOL)
        pickleFile.close()
        j=j+1
    ####load tree
#    pickleFile = open('%s/tree%02d.dat'%(rootdir,0), 'rb')
#    (all_files,tree,wd) = cPickle.load(pickleFile)
#    pickleFile.close()
#    findall(tree,all_files)

