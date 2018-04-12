import time
import numpy as np
from sklearn.neighbors import NearestNeighbors
import maya.cmds as cmds

#Set ICP_TEST to False to setup A & B point sets
#Set ICP_TEST to True to test ICP fit

ICP_TEST=False
ICP_TEST=True


def drange(start, stop, step):
     r = start
     while r < stop:
         yield r
         r += step

def createMaterial(name,color,type):
    cmds.sets(renderable=True, noSurfaceShader=True, empty=True, name=name + 'SG' )
    cmds.shadingNode( type, asShader=True, name=name )
    cmds.setAttr( name+".color", color[0], color[1], color[2], type='double3')
    cmds.connectAttr(name+".outColor", name+"SG.surfaceShader")

def assignMaterial (name, object):
    cmds.sets(object, edit=True, forceElement=name+'SG')

def assignNewMaterial( name, mycolor, type, object):
    createMaterial (name, mycolor, type)
    assignMaterial (name, object)



def best_fit_transform(A, B):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    '''

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
       Vt[m-1,:] *= -1
       R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R,centroid_A.T)

    # homogeneous transformation
    T = np.identity(m+1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t


def nearest_neighbor(src, dst):
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nxm array of points
        dst: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    '''

    assert src.shape == dst.shape
    
    #print 'src',src

    neigh = NearestNeighbors(n_neighbors=1,radius=10.0)
    #print 'neigh',neigh
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()


def icp(A, B, init_pose=None, max_iterations=20, mytolerance=0.001):
    '''
    The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
    Input:
        A: Nxm numpy array of source mD points
        B: Nxm numpy array of destination mD point
        init_pose: (m+1)x(m+1) homogeneous transformation
        max_iterations: exit algorithm after max_iterations
        mytolerance: convergence criteria
    Output:
        T: final homogeneous transformation that maps A on to B
        distances: Euclidean distances (errors) of the nearest neighbor
        i: number of iterations to converge
    '''

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m+1,A.shape[0]))
    dst = np.ones((m+1,B.shape[0]))
    src[:m,:] = np.copy(A.T)
    dst[:m,:] = np.copy(B.T)

    # apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)

    prev_error = 0

    for i in range(max_iterations):
        # find the nearest neighbors between the current source and destination points
        distances, indices = nearest_neighbor(src[:m,:].T, dst[:m,:].T)

        # compute the transformation between the current source and nearest destination points
        T,_,_ = best_fit_transform(src[:m,:].T, dst[:m,indices].T)

        # update the current source
        src = np.dot(T, src)

        # check error
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < mytolerance:
            print 'converged at iteration %d'%(i)
            break
        prev_error = mean_error
        
        
        #Continuously update B set fitting positions        
        # Make C a homogeneous representation of B
        C = np.ones((N, 4))
        C[:,0:3] = np.copy(B)

        # Transform C
        C = np.dot(T, C.T).T
        
        print 'B',B
        print 'C',C[:, 0:3]
        
        print 'C-B',C[:, 0:3] - B
        
        dirArray = C[:, 0:3] - B
       
        
        for dir in dirArray:
            print 'dir =',dir
        
        
        for i in range(len(A)/2):
            print 'A[i]',A[i]
            print 'C[i]',C[i]
                        
            xstart=A[i][0]
            ystart=A[i][1]
            zstart=A[i][2]      
            xend=C[:, 0:3][i][0]
            yend=C[:, 0:3][i][1]
            zend=C[:, 0:3][i][2]
            
            step = ( (xend-xstart)/100.0, (yend-ystart)/100.0, (zend-zstart)/100.0)
            print 'step',step   

            curValx =0
            curValy =0
            curValz =0

            for t in drange(xstart, xend , step[0]):
                curValx = xstart + t * (xend-xstart)
                 
            for t in drange(ystart, yend , step[1]):
                curValy = ystart + t * (yend-ystart)
                
            for t in drange(zstart, zend , step[2]): 
                curValz = zstart + t * (zend-zstart)                       
                           
            
            if cmds.objExists('pSphere'+str(i+N)):
                    #B point set already created
                    print 'pSphere'+str(1+i+N)+' exists!!'
                    print 'pSphere'+str(1+i+N)+' selected!!!'
                     
                    cmds.select('pSphere'+str(1+i+N))
                    cmds.xform(translation=(curValx,curValy,curValz) )
                    cmds.refresh()
                    time.sleep(0.1)
          
            
        '''
        counter=1
        for pos in B:
            print pos
            
            if cmds.objExists('pSphere'+str(counter+N)):
                #B point set already created
                print 'pSphere'+str(counter+N)+' exists!!'
                cmds.select('pSphere'+str(counter+N))
                #cmds.xform(translation=(pos[0],pos[1],pos[2]) )
                counter+=1
        '''     
    cmds.refresh()
    
    # calculate final transformation
    T,_,_ = best_fit_transform(A, src[:m,:].T)

    return T, distances, i
    
    
    
    
'''TEST'''    
import numpy as np
import time
#import icp

# Constants
N = 50                                   # number of random points in the dataset

'''num_tests WAS 100, not one but it doesn't seem to help in converging better, so brought it down to 1 to converge faster'''
num_tests = 1                             # number of test iterations 

dim = 3                                     # number of dimensions of the points
noise_sigma = .01                           # standard deviation error to be added
translation = 2                           # max translation of the test set
rotation = .1                               # max rotation (radians) of the test set


def rotation_matrix(axis, theta):
    axis = axis/np.sqrt(np.dot(axis, axis))
    a = np.cos(theta/2.)
    b, c, d = -axis*np.sin(theta/2.)

    return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                  [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
                  [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c]])


def test_best_fit():

    # Generate a random dataset
    np.random.seed(985)
    A = 150*np.random.rand(N, dim)
    
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
        T, R1, t1 = best_fit_transform(B, A)#icp.best_fit_transform(B, A)
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
    np.random.seed(985)
    A = 50*np.random.rand(N, dim)
    
    cmds.select(all=True)
    cmds.delete()
    print 'A',A
    
    #create A point set
    for pos in A:
        print pos
        s=cmds.polySphere()
        #print s[0]
        assignNewMaterial( 'ballShader'+s[0] , (1, 0, 0), 'blinn', s[0])
        cmds.select(s[0])
        cmds.xform(translation=(pos[0],pos[1],pos[2]) )
        
        
    total_time = 0
    
    

    for i in range(num_tests):

        B = np.copy(A)

        # Translate
        np.random.seed(555)
        t = np.random.rand(dim)*translation
        B += t

        # Rotate
        R = rotation_matrix(np.random.rand(dim), np.random.rand() * rotation)
        B = np.dot(R, B.T).T

        # Add noise
        B += np.random.randn(N, dim) * noise_sigma

        # Shuffle to disrupt correspondence
        np.random.shuffle(B)
        
        #create B point set
        
        
        counter=1
        for pos in B:
            #print pos
            
            if cmds.objExists('pSphere'+str(counter+N)):
                #B point set already created
                #print 'pSphere'+str(counter+N)+' exists!!'
                cmds.select('pSphere'+str(counter+N))
                cmds.xform(translation=(pos[0],pos[1],pos[2]) )
                counter+=1
               
            else:
                
                s=cmds.polySphere()
                #print s[0]
                assignNewMaterial( 'ballShader'+s[0] , (0, 1, 0), 'blinn', s[0])
                cmds.select(s[0])
                cmds.xform(translation=(pos[0],pos[1],pos[2]) )
                counter+=1
                    
                
                
        if ICP_TEST==True:
            
            # Run ICP
            start = time.time()
            T, distances, iterations = icp(B, A, mytolerance=0.000001)#icp.icp(B, A, mytolerance=0.000001)
            total_time += time.time() - start
    
            # Make C a homogeneous representation of B
            C = np.ones((N, 4))
            C[:,0:3] = np.copy(B)
    
            # Transform C
            C = np.dot(T, C.T).T
        
        
            #create B point set, after being transformed to C point set
            
            counter=1        
            for pos in C:
                #print pos
                #s=cmds.polySphere()
                #print s[0]
                #assignNewMaterial( 'ballShader'+s[0] , (0, 1, 0), 'blinn', s[0])
                cmds.select('pSphere'+str(counter+N))
                cmds.xform(translation=(pos[0],pos[1],pos[2]) )
                counter+=1
    
        

            assert np.mean(distances) < 6*noise_sigma                   # mean error should be small
            assert np.allclose(T[0:3,0:3].T, R, atol=6*noise_sigma)     # T and R should be inverses
            assert np.allclose(-T[0:3,3], t, atol=6*noise_sigma)        # T and t should be inverses

    #print('icp time: {:.3}'.format(total_time/num_tests))

    return


if __name__ == "__main__":
    test_best_fit()
    test_icp()