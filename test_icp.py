import sys
sys.path.append('/usr/lib64/python2.7/site-packages/')
sys.path.append('/home/yioannidis/Downloads/Maya_scripts/Dr. XiaosongYang/ICP-TEST')

#sys.path.remove('/home/yioannidis/Desktop/testPipeline/testPipelineCervical')
#sys.path.remove('/home/yioannidis/Desktop/testPipeline/testPipelineCervical/testICPCUBE')
import maya.cmds as cmds

import numpy as np
import time
import icp
reload(icp)

import math as math

def createMaterial(name,color,type):
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
N = 4                                    # number of random points in the dataset
num_tests = 100                             # number of test iterations
dim = 3                                     # number of dimensions of the points
noise_sigma = .1                           # standard deviation error to be added
translation = 1                            # max translation of the test set
rotation = 1                               # max rotation (radians) of the test set


def rotation_matrix(axis, theta):
    axis = axis/np.sqrt(np.dot(axis, axis))
    a = np.cos(theta/2.)
    b, c, d = -axis*np.sin(theta/2.)

    return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                  [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
                  [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c]])





cmds.file(force=True, new=True)
cmds.select(all=True)
cmds.delete()
import maya.cmds as cmds
sparseCoeff=10

def test_best_fit():
    
    # Generate a random dataset
    A = sparseCoeff*np.random.rand(N, dim)
    
    total_time = 0

    for i in range(num_tests):

        B = np.copy(A)

        # Translate
        t = np.random.rand(dim)*translation
        B += t

        # Rotate
        R = rotation_matrix(np.random.rand(dim), np.random.rand()*rotation)
        B = np.dot(R, B.T).T

        # Add noise
        B += np.random.randn(N, dim) * noise_sigma
        

        # Find best fit transform
        start = time.time()
        T, R1, t1 = icp.best_fit_transform(B, A)
        total_time += time.time() - start

        # Make C a homogeneous representation of B
        C = np.ones((N, 4))
        C[:,0:3] = B

        # Transform C
        C = np.dot(T, C.T).T

        assert np.allclose(C[:,0:3], A, atol=6*noise_sigma) # T should transform B (or C) to A
        assert np.allclose(-t1, t, atol=6*noise_sigma)      # t and t1 should be inverses
        assert np.allclose(R1.T, R, atol=6*noise_sigma)     # R and R1 should be inverses

    print('best fit time: {:.3}'.format(total_time/num_tests))

    return


def test_icp():

    # Generate a random dataset
    A = sparseCoeff*np.random.rand(N, dim)
    counter=0
    for i in A:
        counter+=1
        s=cmds.polySphere()
        cmds.move(i[0],i[1],i[2])
        s1color=[0,1,0]
        assignNewMaterial( 'ballShader' + str(counter), (s1color[0], s1color[1], s1color[2]), 'blinn', s[0] )
    

    total_time = 0
    
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
        
      
        
    for i in B:  
        counter+=1          
        s=cmds.polySphere()
        cmds.move(i[0],i[1],i[2])
        print 'move,'+str(s[0])+' to ',i[0],i[1],i[2]
        s2color=[1,0,0]
        assignNewMaterial( 'ballShader' + str(counter), (s2color[0], s2color[1], s2color[2]), 'blinn', s[0] )
        
    print 'B',B  
    # Run ICP
    start = time.time()
    T, distances, iterations = icp.icp(B, A, tolerance=0.000001)
    total_time += time.time() - start

    # Make C a homogeneous representation of B
    C = np.ones((N, 4))
    C[:,0:3] = np.copy(B)

    # Transform C
    C = np.dot(T, C.T).T
    
           
    for i in range((N+1),(2*N+1),1):
        cmds.select('pSphere'+str(i))
        cmds.refresh()
        time.sleep(0.5)
        print 'pSphere'+str(i)
        cmds.move(C[i-(N+1)][0],C[i-(N+1)][1],C[i-(N+1)][2])    
        cmds.refresh()
        time.sleep(0.5)

    assert np.mean(distances) < 6*noise_sigma                   # mean error should be small
    assert np.allclose(T[0:3,0:3].T, R, atol=6*noise_sigma)     # T and R should be inverses
    assert np.allclose(-T[0:3,3], t, atol=6*noise_sigma)        # T and t should be inverses

    print('icp time: {:.3}'.format(total_time/num_tests))

    return


if __name__ == "__main__":
    test_best_fit()
    test_icp()