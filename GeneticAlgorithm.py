class genom:

    genom_list = None
    evaluation = None
    distance = None

    def __init__(self, genom_list, evaluation, distance):
        self.genom_list = genom_list
        self.evaluation = evaluation
        self.distance = distance


    def getGenom(self):
        return self.genom_list

    def getEvaluation(self):
        return self.evaluation

    def getDistance(self):
        return self.distance

    def setGenom(self, genom_list):
        self.genom_list = genom_list

    def setEvaluation(self, evaluation):
        self.evaluation = evaluation

    def setDistance(self, distance):
        self.distance = distance
