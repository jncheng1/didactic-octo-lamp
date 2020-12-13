from scipy.stats import entropy
import math
import numpy
import statistics

# Function Reads in the data from file and constructs
# a list of feature vectors
def dataReader(filename):
    
    # Opens the data file
    f = open(filename, 'r')
    data_file = f.readlines()
    dataList = []

    # Reads data from file
    for line in data_file:
        if not line:
            break
        
        feature_vector = line.strip().split(' ')
        label = feature_vector[len(feature_vector)-1]
        feature_vector.pop()

        # Converts data into float values
        feature_vector = numpy.array(feature_vector)
        feature_vector = numpy.asarray(feature_vector,float)
        label = float(label)
        
        # Structures the data into pairs (feature vector, label)
        data = (feature_vector, label)
        dataList.append(data)

    f.close()

    return dataList

# Calculates the entropy for the set
def getEntropy(feature):

    label_one = 0.0
    label_zero = 0.0

    for i in range(0,len(feature)):
        if (feature[i][1] == 1.0):
            label_one += 1
        else:
            label_zero += 1

    total_label = label_one + label_zero
    return entropy([label_one/total_label,label_zero/total_label])

# Calculates the Information Gain for the feature
def getIG(feature, index):

    feature.sort(key = lambda x :x[0][index])
    dataList =[]
    set_entropy = getEntropy(feature)
    # Calculates all entropy values and finds the smallest
    for i in range(0, len(feature)-1):
        if(feature[i][0][index] != feature[i+1][0][index]):
            midpoint = (feature[i][0][index] + feature[i+1][0][index])/2
            entropy = ((i+1)/len(feature))*(getEntropy(feature[0:i+1])) + ((len(feature)-(i+1))/len(feature))*getEntropy(feature[i+1:len(feature)])
            data = (set_entropy-entropy,midpoint,i,index)
            dataList.append(data)
        else:
            data = (-1, 0,0,0)
            dataList.append(data)

    return max(dataList, key= lambda x: x[0]) 

# Checks to see if all nodes have the same label
def sameLabelAll(feature):
    
    label_one = 0.0
    label_zero = 0.0

    for i in range(0,len(feature)):
        if (feature[i][1] == 1.0):
            label_one += 1
        else:
            label_zero += 1
    
    if(label_zero == 0):
        return True
    
    if(label_one == 0):
        return True
    
    else:
        return False

# Gets the majority label
def getMajority(feature):
    
    label_one = 0.0
    label_zero = 0.0

    for i in range(0,len(feature)):
        if (feature[i][1] == 1.0):
            label_one += 1
        else:
            label_zero += 1
    
    if(label_zero < label_one):
        return 1
    else:
        return 0
    
# Calculates the Information Gain for each feature 
# so that the optimal feature can be selected
def selectSplit(dataList):
    splits = []

    # Gets the information gain for every feature
    for i in range(0, len(dataList[0][0])):
        splits.append(getIG(dataList,i))

    # Selects the highest information gain value
    return max(splits, key= lambda x: x[0])

# Nodes used to populate tree
class Node:

    # Initializes node
    def __init__(self,vectors,midpoint,feature,level):
        self.left = None
        self.right = None
        self.vectors = vectors
        self.midpoint = midpoint
        self.feature = feature
        self.level = level
    
    # prints out information for node
    def printRule(self):

        if(self.feature == -1):
            print("This is a Leaf Node with prediction:" + str(self.midpoint) + " Size:" + str(len(self.vectors)) + " Level:" + str(self.level))
        else:
            print("Feature Vector " +str(self.feature+1) + " < " + str(self.midpoint) + " Size:" + str(len(self.vectors)) + " Level:" + str(self.level) )

    # Inserts elements 
    def insert(self, dataList, side, level):

        #right insert
        if(side == 0):
            
            #Checks for leaf nodes
            if(sameLabelAll(dataList)):
                self.right = Node(dataList, dataList[0][1], -1,level)
                return None
            if(len(dataList) == 1):
                self.right = Node(dataList, dataList[0][1], -1,level)
                return None
            #Checks for Empty
            if(len(dataList) == 0):
                return None

            #find node
            split = selectSplit(dataList)
            dataList.sort(key = lambda x :x[0][split[3]])

            #creates node
            self.right = Node(dataList, split[1], split[3],level)

            #Splits the tree 
            leftList = dataList[:split[2]+1]
            rightList = dataList[(split[2]+1)-len(dataList):]

            #Recursively Populates the tree
            self.right.insert(leftList, 1,level+1)
            self.right.insert(rightList, 0,level+1)

        #left insert
        else:
            
            #Checks for leaf nodes
            if(sameLabelAll(dataList)):
                self.left = Node(dataList, dataList[0][1], -1, level)
                return None
            if(len(dataList) == 1):
                self.left = Node(dataList, dataList[0][1], -1,level)
                return None
            #Checks for Empty
            if(len(dataList) == 0):
                return None

            #find root node
            split = selectSplit(dataList)
            dataList.sort(key = lambda x :x[0][split[3]])

            #creates node
            self.left = Node(dataList, split[1], split[3],level)

            #Splits the tree 
            leftList = dataList[:split[2]+1]
            rightList = dataList[(split[2]+1)-len(dataList):]

            #Recursively Populates the tree
            self.left.insert(leftList, 1,level+1)
            self.left.insert(rightList, 0,level+1)
    
    # Prints out nodes in pre order traversal
    def print(self):
        
        self.printRule()
        
        if(self.left != None):
            self.left.print()

        if(self.right != None):
            self.right.print()

    # Searches for label
    def search(self, v1):
        
        #Returns label the label
        if(self.feature == -1):
            return self.midpoint

        #search left
        if(v1[self.feature] <= self.midpoint):
            return self.left.search(v1)
        
        #search right
        else:
            return self.right.search(v1)
        
# Tree class for constructing ID3decision tree
class Tree:

    # Initializes the tree
    def __init__ (self):
        self.root = None
        self.size = 0
    
    # Return the root of Tree
    def getRoot(self):
        return self.root    
    
    # Populates the tree
    def buildID3(self, dataList):

        #Checks to make sure that all labels are different
        if(sameLabelAll(dataList)):
            self.root = Node(dataList, dataList[0][1], -1, 0)
            return None

        #find root node
        split = selectSplit(dataList)
        #sorts data by the select feature from split
        dataList.sort(key = lambda x :x[0][split[3]])
        
        self.root = Node(dataList, split[1], split[3],0)

        #Splits the data
        leftList = dataList[:split[2]+1]
        rightList = dataList[(split[2]+1)-len(dataList):]

        #Recursively Populates the tree
        self.root.insert(leftList, 1, 1)
        self.root.insert(rightList, 0, 1)
    
    # Print the tree using inorder traversel
    def printTree(self):

        self.getRoot().print()
    
    # Searches for label
    def search(self, vector):

        return self.root.search(vector)

    # Calculate Percent Error
    def calcError(self, dataList):

        incorrect = 0

        # Loops through all vectors
        for i in range(0, len(dataList)):

            vector = dataList[i][0]
            label = dataList[i][1]
            prediction = self.search(vector)

            if(prediction != label):
                incorrect +=1

        return incorrect/len(dataList)

    # Prunes the Tree to avoid overfitting
    def pruning(self, testList, validationList):

        percentError = self.calcError(validationList)
        print("Validation Error: "+str(percentError))
        pruneError = 0
        i = 0
        #visited = []
        queue = []

        #visited.append(self.getRoot())
        queue.append(self.getRoot())
    
        #Attempts Pruning Nodes through BFS exploration
        while queue:
            ptr = queue.pop(0)
            label = getMajority(ptr.vectors)
            i+=1
            #Saves the current node
            currFeature = ptr.feature
            currMidPt = ptr.midpoint

            #Attemp Pruning a Node
            ptr.feature = -1
            ptr.midpoint = label
            pruneError = self.calcError(validationList)
            print("Prune Attempt"+ str(i) + ": "+ str(pruneError))

            #Tree will be pruned
            if(pruneError < percentError):
                percentError = pruneError
                testError = self.calcError(testList)
                print("Test Error After Pruning Success" + str(testError))
            
            #Tree should not be pruned
            else:

                #Unprunes the Tree
                ptr.feature = currFeature
                ptr.midpoint = currMidPt
                
                #Stores the nodes to visit next
                if(ptr.left != None):
                    queue.append(ptr.left)
                if(ptr.right != None):
                    queue.append(ptr.right)


# Runs the Program with Pruning
dataList = dataReader("pa2train.txt")
testList = dataReader("pa2test.txt")
validationList = dataReader("pa2validation.txt")
id3tree = Tree()
id3tree.buildID3(dataList)
print("Training Error:" + str(id3tree.calcError(dataList)))
print("Test Error:" + str(id3tree.calcError(testList)))
id3tree.pruning(testList,validationList)


