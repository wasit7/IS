# -*- coding: utf-8 -*-
"""
Created on Tue Dec 02 16:06:41 2014

@author: Wasit
"""

from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import kdtree
import cPickle
import nn
import os
def findall(tree,all_files):
    for f in all_files:
        fsg=nn.getHash3(f)
        d,ids=tree.query(fsg,k=10)
        plt.figure(1)  
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

def findall2(all_trees,all_files):
    #show 10 candidates    
    k=10    
    for files in all_files:
        for f in files:
            temp={'ds':[],'ids':[],'files':[]}
            fsg=nn.getHash3(f)
            for tree in all_trees:
                ds,ids=tree.query(fsg,k)
                temp['ds'].append(ds)
                temp['ids'].append(ids)
                temp['files'].append(files) 
            
            id_sort=np.argsort(temp['ds'])
            sort_d=temp['ds'][id_sort[0:k]]
            sort_ids=temp['ids'][id_sort[0:k]]
            sort_files=temp['files'][id_sort[0:k]]
            plt.figure(1)  
            for i in xrange(k):
                print "%03d dis: %.3f %s"%(sort_ids[i],sort_d[i],sort_files[i])
                plt.subplot(2,5,i+1)
                plt.title("%03d dis: %.3f"%(sort_ids[i],sort_d[i]),fontsize=10)
                im_icon=np.array(Image.open(sort_files[i]).convert('L'))
                plt.imshow(im_icon)
                plt.axis('off')
            plt.set_cmap('gray')    
            plt.show()        
            plt.ginput(1)
            
if __name__ == '__main__':
    # patch module-level attribute to enable pickle to work
    kdtree.node = kdtree.KDTree.node
    kdtree.leafnode = kdtree.KDTree.leafnode
    kdtree.innernode = kdtree.KDTree.innernode
    
    ####load tree
    all_treesfiles=[]
    for root, dirs, files in os.walk(nn.rootdir):
        for f in files:
                if f.endswith('dat'):
                    all_treesfiles.append(os.path.join(root,f))
    
    all_trees=[]
    all_files=[]
    for tf in all_treesfiles:
        pickleFile = open(tf, 'rb')
        files,tree,wd = cPickle.load(pickleFile)
        pickleFile.close()
        all_trees.append(tree)
        all_files.append(files)
        
        
    #findall2(all_trees,all_files)
    #make list of query files
    q_files=[]
    for root, dirs, files in os.walk('query'):
        for f in files:
                if f.endswith('jpg') or f.endswith('JPG'):
                    q_files.append(os.path.join(root,f))
    #the number of candidates                
    k=10    
    for f in q_files:
        #plt.figure(1)
        #plt.imshow(Image.open(f).convert('L'))
        temp={'ds':[],'ids':[],'files':[]}
        fsg=nn.getHash3(f)
        for sub in xrange(len(all_trees)):
            ds,ids=all_trees[sub].query(fsg,k)
            for i in xrange(k):
                temp['ds'].append(ds[i])
                temp['ids'].append(ids[i])
                temp['files'].append(all_files[sub][ids[i]])

        id_sort=np.argsort(temp['ds'])
        sort_d=np.array(temp['ds'])[id_sort[0:k]]
        sort_ids=np.array(temp['ids'])[id_sort[0:k]]
        sort_files=[]
        for i in xrange(k):
            sort_files.append(temp['files'][id_sort[i]])
        print "\n"
        for i in xrange(k):
            print "%03d dis: %.3f %s"%(sort_ids[i],sort_d[i],sort_files[i])
            
#display        
#        plt.figure(2) 
#        plt.cla()
#        for i in xrange(k):
#            print "%03d dis: %.3f %s"%(sort_ids[i],sort_d[i],sort_files[i])
#            plt.subplot(2,5,i+1)
#            plt.title("%03d dis: %.3f"%(sort_ids[i],sort_d[i]),fontsize=10)
#            plt.xlabel('%s'%sort_files[i])            
#            im_icon=np.array(Image.open(sort_files[i]).convert('L'))
#            plt.imshow(im_icon)
#            plt.axis('off')
#        plt.set_cmap('gray')    
#        plt.show()        
#        plt.ginput(1)