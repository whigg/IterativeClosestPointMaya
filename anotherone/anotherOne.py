
#select 'PROJECTED3DTO2D' spheres
PROJECTED3DTO2D=cmds.ls(selection=True)
len(PROJECTED3DTO2D)
B=[]    
for i in PROJECTED3DTO2D:
    cmds.select(i)
    curPointPosition = cmds.xform( i, query=True, translation=True, worldSpace=True )
    #cmds.xform(i,translation=(curPointPosition[0],curPointPosition[1],100),worldSpace=True  )
    B.append(cmds.xform( i, query=True, translation=True, worldSpace=True ))

len(B)    
    


#select '_2D_read_from_file' spheres
cmds.select(clear=True)

_2D_read_from_file=cmds.ls(selection=True)
A=[]
for i in _2D_read_from_file:
    A.append(cmds.xform( i, query=True, translation=True, worldSpace=True ))
len(A)




import sys
sys.path.append('/usr/lib64/python2.7/site-packages/')
sys.path.append('/home/yioannidis/Downloads/Maya_scripts/Dr. XiaosongYang/ICP-TEST/anotherone')
#sys.path.remove('/home/yioannidis/Desktop/testPipeline/testPipelineCervical')
#sys.path.remove('/home/yioannidis/Downloads/Maya_scripts/Dr. XiaosongYang/ICP-TEST')
import maya.cmds as cmds

import numpy as np
import time
import icp
reload(icp)


import math as math

def createMaterial(name,color,type):
    if not cmds.objExists(name):
        cmds.sets(renderable=True, noSurfaceShader=True, empty=True, name=name + 'SG' )
        cmds.shadingNode( type, asShader=True, name=name )
        cmds.setAttr( name+".color", color[0], color[1], color[2], type='double3')
        cmds.connectAttr(name+".outColor", name+"SG.surfaceShader")

def assignMaterial (name, object):
    cmds.sets(object, edit=True, forceElement=name+'SG')


def assignNewMaterial( name, color, type, object):
    createMaterial (name, color, type)
    assignMaterial (name, object)


# Constants
N = len(_2D_read_from_file)                 # number of random points in the dataset
num_tests = 100                             # number of test iterations
dim = 3                                     # number of dimensions of the points
noise_sigma = .01                           # standard deviation error to be added
translation = .01                            # max translation of the test set
rotation = .01                               # max rotation (radians) of the test set


def rotation_matrix(axis, theta):
    axis = axis/np.sqrt(np.dot(axis, axis))
    a = np.cos(theta/2.)
    b, c, d = -axis*np.sin(theta/2.)

    return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                  [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
                  [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c]])




'''
cmds.file(force=True, new=True)
cmds.select(all=True)
cmds.delete()
'''
import maya.cmds as cmds
sparseCoeff=10


import icp
reload(icp)

def test_icp(A,B):

    # Generate a random dataset
    #A = sparseCoeff*np.random.rand(N, dim)
    counter=0
    for i in A:    
        cmds.select(_2D_read_from_file[counter])
        s1color=[0,1,0]
        assignNewMaterial( 'ballShader' + str(counter), (s1color[0], s1color[1], s1color[2]), 'blinn', _2D_read_from_file[counter] )
        counter+=1


    total_time = 0
    
    '''
    B = np.copy(A)

    # Translate
    t = np.random.rand(dim)*translation
    B += t

    # Rotate
    R = rotation_matrix(np.random.rand(dim), np.random.rand() * rotation)
    B = np.dot(R, B.T).T

    # Add noise
    B += np.random.randn(N, dim) * noise_sigma

    # Shuffle to disrupt correspondence
    np.random.shuffle(B)
    '''
    
    
    '''
    for i in range(num_tests):

        B = np.copy(A)

        # Translate
        t = np.random.rand(dim)*translation
        B += t

        # Rotate
        R = rotation_matrix(np.random.rand(dim), np.random.rand() * rotation)
        B = np.dot(R, B.T).T

        # Add noise
        B += np.random.randn(N, dim) * noise_sigma

        # Shuffle to disrupt correspondence
        np.random.shuffle(B)
        
    '''  
    
    for i in B:  
        cmds.select(PROJECTED3DTO2D[counter-len(PROJECTED3DTO2D)])
        s2color=[1,0,0]
        assignNewMaterial( 'ballShader' + str(counter), (s2color[0], s2color[1], s2color[2]), 'blinn', PROJECTED3DTO2D[counter-len(PROJECTED3DTO2D)] )
        counter+=1          

        
        
    A=np.asarray(A)
    B=np.asarray(B)
    # Run ICP
    start = time.time()
    T, distances, iterations = icp.icp(B, A, PROJECTED3DTO2D,tolerance=0.001 )
    total_time += time.time() - start

    # Make C a homogeneous representation of B
    C = np.ones((N, 4))
    C[:,0:3] = np.copy(B)

    # Transform C
    C = np.dot(T, C.T).T
    exit()
    counter=0       
    for i in range(len(PROJECTED3DTO2D)):
        cmds.select(PROJECTED3DTO2D[counter])        
        #time.sleep(0.001)
        print PROJECTED3DTO2D[counter]
        cmds.move(C[i][0],C[i][1],C[i][2])    
        cmds.refresh()
        #time.sleep(0.001)
        counter+=1
        

    #assert np.mean(distances) < 6*noise_sigma                   # mean error should be small
    #assert np.allclose(T[0:3,0:3].T, R, atol=6*noise_sigma)     # T and R should be inverses
    #assert np.allclose(-T[0:3,3], t, atol=6*noise_sigma)        # T and t should be inverses

    print('icp time: {:.3}'.format(total_time/num_tests))

    return

reload(icp)
test_icp(A,B)