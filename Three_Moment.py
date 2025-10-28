class Node():
    def __init__(self, x, y):   #the class node stores various points on the structure
        self.x = x
        self.y = y
class Support():
    def __init__(self, node, supporttype):
        self.node = node
        self.supporttype = supporttype
class Pointload():
    def __init__(self, node, magnitude):
        self.node = node
        self.magnitude = magnitude
class Span():
    def __init__(self, leftSupport, rightSupport, listPL, listUDL):
        self.leftSupport = leftSupport
        self.rightSupport = rightSupport
        self.listPL = listPL
        self.listUDL = listUDL
    def length(self):
        x1 = self.leftSupport.node.x
        x2 = self.rightSupport.node.x
        self.L = x2 - x1
 
        
class UDL():
    def __init__(self, startSupport, endSupport, intensity):
        self.startSupport = startSupport
        self.endSupport = endSupport
        self.intensity = intensity
class TMSpan():                    #TMSpan means three moment span 
    def __init__(self, leftSpan, rightSpan):
        self.leftSpan = leftSpan
        self.rightSpan = rightSpan

A = Node(0,0)
E = Node(2,0)
B = Node(6,0)
F = Node(8,0)
C = Node(11,0)
D = Node(15,0)

supportA = Support(A, 'pin')
supportB = Support(B, 'roller')
supportC = Support(C, 'roller')
supportD = Support(D, 'roller')

udl1 = UDL(supportC, supportD, 3)

PloadE = Pointload(E, 9)
PloadF = Pointload(F, 8)

span1 = Span(supportA, supportB, [PloadE], [])
span2 = Span(supportB, supportC, [PloadF], [])
span3 = Span(supportC, supportD, [], [udl1])   


threeMomentSpan1 = TMSpan(span1, span2)
threeMomentSpan2 = TMSpan(span2, span3)

