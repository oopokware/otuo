import numpy as np

class joint():
    def __init__(self,x,y):
        self.xcoord=x
        self.ycoord=y

j=int(input('Please enter number of joints: '))
m=int(input('Please enter number of members: '))
if (2*j-m)==3:
    print('Please Input {} joints \n'.format(j))
else:
    print('not statically determinate! Try Again!!!')
        
class solver():
    def __init__(self,joint1,joint2):
        self.joint1=joint1
        self.joint2=joint2
    def length(self):
        x1=self.joint1.xcoord
        x2=self.joint2.xcoord
        y1=self.joint1.ycoord
        y2=self.joint2.ycoord
        L=np.sqrt((x2-x1)**2+(y2-y1)**2)
        return L
    def directionCosine(self):
        x1=self.joint1.xcoord
        x2=self.joint2.xcoord
        y1=self.joint1.ycoord
        y2=self.joint2.ycoord
        L=np.sqrt((x2-x1)**2+(y2-y1)**2)
        cos=((x2-x1)/L)
        sine=((y2-y1)/L)
        return cos, sine
        
class load():
    def __init__(self,joint,loadType,value,direction):  #loadType is either vertical or horizontal
        self.loadJoint=joint              #direction is either downwards or upwards for vertical loads
        self.loadType=loadType            #direction is either left or right for horizontal loads
        self.Ldirect=direction
        self.value=value
        
class support():
    def __init__(self,SJoint1,SJoint2,supportType,reactionType,load): #reactionType is either vertical or horizontal
        self.SJoint1=SJoint1
        self.SJoint2=SJoint2
        self.support=supportType
        self.reactionType=reactionType
        self.load=load
        
    def reaction(self):
        xa = self.SJoint1.xcoord
        xb = self.SJoint2.xcoord
        ya = self.SJoint1.ycoord
        yb = self.SJoint2.ycoord
        Ls = np.sqrt((xb - xa)**2 + (yb - ya)**2)  # Length between supports

        # Calculate the direction cosines of the member
        cos, sin = self.load.directionCosine()

        # Calculate the length of the member from the load to joint B
        LL = np.sqrt((xb - self.load.loadJoint.xcoord)**2 + (yb - self.load.loadJoint.ycoord)**2)

        # Calculate the horizontal and vertical components of the load
        H = self.load.value * cos
        V = self.load.value * sin

        # Calculate the reaction forces using force equilibrium equations
        if self.reactionType == "vertical":
            if self.support == "roller":
                R1 = 0
                R2 = V
            elif self.support == "pin":
                R1 = V * (xb - xa) / Ls
                R2 = V - R1
        elif self.reactionType == "horizontal":
            if self.support == "roller":
                R1 = H
                R2 = 0
            elif self.support == "pin":
                R1 = H * (yb - ya) / Ls
                R2 = H - R1

        # Return the reaction forces
        return R1, R2
class truss():
    def __init__(self, joints, members, supports, loads):
        self.joints = joints
        self.members = members
        self.supports = supports
        self.loads = loads
        
    def solve(self):
        # Initialize reaction forces to zero
        R = np.zeros((2, len(self.supports)))

        # Iterate through each joint in the truss system
        for i, joint in enumerate(self.joints):
            # Initialize the forces acting on the joint
            F = np.zeros((2, 1))

            # Iterate through each member connected to the joint
            for member in self.members:
                # Check if the member is connected to the current joint
                if joint in [member.joint1, member.joint2]:
                    # Calculate the direction cosine of the member
                    cos, sin = member.directionCosine()

                    # Determine which joint is not the current joint
                    other_joint = member.joint2 if member.joint1 == joint else member.joint1

                    # Iterate through each load connected to the other joint
                    for load in self.loads:
                        if load.loadJoint == other_joint:
                            # Calculate the horizontal and vertical components of the load
                            H = load.value * cos
                            V = load.value * sin

                            # Add the load forces to the joint forces
                            if load.Ldirect == "upwards":
                                F[1][0] -= V
                            elif load.Ldirect == "downwards":
                                F[1][0] += V
                            elif load.Ldirect == "left":
                                F[0][0] -= H
                            elif load.Ldirect == "right":
                                F[0][0] += H

                    # Add the member forces to the joint forces
                    F += member.force()

            # Solve for the unknown forces at the joint using the principle of equilibrium
            if i == 0:
                A = np.array([[1, 0], [0, 1]])
                b = F
            else:
                A = np.vstack((A, [cos, sin]))
                b = np.vstack((b, F))

            if len(A) == len(self.joints):
                # Solve for the unknown forces using matrix algebra
                x = np.linalg

   

        

