class genom:

    genom_list = None
    evaluation = None
    edge = None

    def __init__(self, genom_list, evaluation, edge):
        self.genom_list = genom_list
        self.evaluation = evaluation
        self.edge = edge


    def getGenom(self):
        return self.genom_list

    def getEvaluation(self):
        return self.evaluation

    def getEdge(self):
        return self.edge

    def setGenom(self, genom_list):
        self.genom_list = genom_list

    def setEvaluation(self, evaluation):
        self.evaluation = evaluation

    def setEdge(self, edge):
        self.edge = edge
