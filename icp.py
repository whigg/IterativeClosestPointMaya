import maya.cmds as cmds

if cmds.objExists('FEMUR*'):
    cmds.select('FEMUR*')  
    cmds.delete()

    
    
cmds.file('/home/yioannidis/3dOrthographicProjection/vpilm07mcj-Femur/' + 'FEMUR-part.obj', i=True)

#set backcafaceculling to full ! Need to be in the front view camera
cmds.setAttr('FEMURShape.backfaceCulling',3)

input('select manually vertices that are not back-face culled in FRONT VIEW in maya..and then run the part 2 of the script')
selectedFrontFacingVertices = cmds.ls(selection=True)

#part 2 START

# http://www.fevrierdorian.com/blog/post/2011/09/27/Quickly-retrieve-vertex-positions-of-a-Maya-mesh-(English-Translation)
def getVtxPos( shapeNode ) :
 
     vtxWorldPosition = []    # will contain positions un space of all object vertex
 
     vtxIndexList = cmds.getAttr( shapeNode+".vrts", multiIndices=True )
 
     for i in vtxIndexList :
          curPointPosition = cmds.xform( str(shapeNode)+".pnts["+str(i)+"]", query=True, translation=True, worldSpace=True )    # [1.1269192869360154, 4.5408735275268555, 1.3387055339628269]
          vtxWorldPosition.append( curPointPosition )
 
     return vtxWorldPosition

if cmds.objExists('pCube1'):
  cmds.select('pCube1')
  cmds.delete()
else:
  print("Warning: no pCube1 exists.")

   
cube=cmds.polyCube()


femur=cmds.ls( 'FEMUR*' )
obj=femur

#cubeVertices = getVtxPos(obj[0])
cubeVertices=selectedFrontFacingVertices

projectedPoints = []
cx, cz = 0,0
sx,sz = 1,1
for i in range(len(cubeVertices)):
    
    #if no ':' symbol is found that means the list has only 1 element    
    if cubeVertices[i].find(':') != -1 :
        print cubeVertices[i]
        
        firstindex = cubeVertices[i].find('[') + 1
        colonindex = cubeVertices[i].find(':')         
        lastindex = cubeVertices[i].find(']') - 1
        
        startnumber=""
        for j in range(int(firstindex), int(colonindex), 1):
            startnumber += str(cubeVertices[i][j])
            print startnumber
                    
        endnumber=""
        for j in range(int(colonindex)+1, int(lastindex)+1, 1):
            endnumber += str(cubeVertices[i][j])
            print endnumber
        
        
        
        for j in range(int(startnumber), int(endnumber)+1, 1):
            print 'FEMUR.vtx['+str(j)+']'
       
                
            ax,ay,az = cmds.xform( 'FEMUR.vtx['+str(j)+']', query=True, translation=True, worldSpace=True )
            print ax,ay,az
            
            #https://en.wikipedia.org/wiki/3D_projection
            bx = sx * ax + cx
            by = sz * ay + cz
            
            id=j
            projectedPoints.append((bx,by,id))
    else:    
        print cubeVertices[i]
        
        firstindex = cubeVertices[i].find('[') + 1
        lastindex = cubeVertices[i].find(']') 
        
        vertNumber=""
        for j in range(int(firstindex), int(lastindex), 1):
            vertNumber += str(cubeVertices[i][j])
            print vertNumber
        
        print 'FEMUR.vtx['+str(vertNumber)+']'  
        ax,ay,az = cmds.xform( 'FEMUR.vtx['+str(vertNumber)+']', query=True, translation=True, worldSpace=True )
        print ax,ay,az
        
        #ax,ay,az =cubeVertices[i][]
        
        #https://en.wikipedia.org/wiki/3D_projection
        bx = sx * ax + cx
        by = sz * ay + cz
        
        id=vertNumber
        projectedPoints.append((bx,by,id))
    
for i in range(1,len(projectedPoints)+1):
    if cmds.objExists('pSphere'+str(i)):
      cmds.select('pSphere'+str(i))
      cmds.delete()
     
#create silhouette 
for i in range(len(projectedPoints)):      
    cmds.polySphere(n='pSphere'+str(projectedPoints[i][2]))
    cmds.xform(translation=(projectedPoints[i][0],projectedPoints[i][1], -50), scale=(0.2, 0.2, 0.2) )

#show silhouette extraced in orthographic project


'''select and show projected points'''
#cmds.select('pSphere*')


'''MANUALY TRANSLATE projected points'''

#Now create a associative relationship/map between each 3D&2D point, so as when I apply a transformation to one the other is also affected

#find difference between previous projectedPoints and currentProjectedPoints
currentProjectedPoints=[]
for i in range(len(projectedPoints)):      
    #print cmds.xform('pSphere'+str(i), translation=True, query=True)
    currentProjectedPoints.append(cmds.xform('pSphere'+str(projectedPoints[i][2]), translation=True, query=True))

traslationUpdateVector = []    
for i in range(len(currentProjectedPoints)):
    if (currentProjectedPoints[i][0]-projectedPoints[i][0] , currentProjectedPoints[i][1]-projectedPoints[i][1]) != (0,0):
        print 'difference in translation between currentProjectedPoints and projectedPoints =', (currentProjectedPoints[i][0]-projectedPoints[i][0] , currentProjectedPoints[i][1]-projectedPoints[i][1])
    
    traslationUpdateVector.append( (currentProjectedPoints[i][0]-projectedPoints[i][0] , currentProjectedPoints[i][1]-projectedPoints[i][1], projectedPoints[i][2]) )

print traslationUpdateVector

for i in range(len(currentProjectedPoints)):
    vertexId=int(traslationUpdateVector[i][2])
    cmds.select('FEMUR.vtx['+str(vertexId)+']')

    print 'FEMUR.vtx['+str(vertexId)+']'
    
    #look up and return the element of which the specified vertexID
    for i in range(len(traslationUpdateVector)):
        print traslationUpdateVector[i]
        if traslationUpdateVector[i][2] == vertexId:
            matchedEntry = traslationUpdateVector[i]
            break;
    
    print matchedEntry
    cmds.xform(r=True, translation=(matchedEntry[0], matchedEntry[1], 0))
    
 