from functools import reduce
from graphviz import Digraph

class NFA: 
	"""Classe para o Automota Finto Não-Determinístico"""
	def __init__(self, automato, estadoInicial, estadosFinais):
		self.auto = automato	
		self.q0 = estadoInicial
		self.F = set(estadosFinais)

	def autoEst(self, state, cadeia):
		"""Retorna vazio caso a transição não esteja definida"""
		states = set([state])
		for a in cadeia: 
			newStates = set([])
			for state in states: 
				try: 
					newStates = newStates | self.auto[state][a]
				except KeyError: pass
			states = newStates
		return states

	def inLanguage(self, cadeia):
		return len(self.autoEst(self.q0, cadeia) & self.F) > 0

	def alphabet(self):
		"""Retorna o alfabeto"""
		Alfabeto = reduce(lambda a,b:set(a)|set(b), [x.keys() for x in N.auto.values()])
		return Alfabeto

	def states(self):
		"""Retorna os estados do NFA"""
		Q = set([self.q0]) | set(self.auto.keys()) | reduce(lambda a,b:a|b, reduce(lambda a,b:a+b, [x.values() for x in self.auto.values()]))	# {q0, all states with outgoing arrows, all with incoming arrows}
		return Q

class DFA():	
    """Classe do Automato Finito Deterministico."""
    def __init__(self, automato, estadoInicial, estadosFinais):
        self.auto = automato
        self.q0 = estadoInicial
        self.F = estadosFinais
        self.graph = Digraph(comment='DFA')
    
    def autoEst(self, state, cadeia):
        # Define a função de transição estendida (retorna o estado P onde o DFA vai estar após ler a cadeia)
        for a in cadeia: 
            state = self.auto[state][a]
        return state
        
    def inLanguage(self, cadeia):
        return self.autoEst(self.q0, cadeia) in self.F

    def generateFormat(self):
        for origem, caminho in self.auto.items():
            print(origem)
            if type(origem) is not frozenset:
                if origem in self.F:
                    self.graph.node(origem,origem, shape='doublecircle')
                else: self.graph.node(origem,origem, shape='circle')
            else:
                m = []
                for i in origem:
                    m.append(i)
                orig = ''.join(sorted(m))
                if origem in self.F:
                    self.graph.node(orig,orig, shape='doublecircle')
                else: self.graph.node(orig,orig, shape='circle')
        for origem, caminho in self.auto.items():
            for transicao,destino in self.auto[origem].items():
                if type(origem) is not frozenset and type(destino) is not frozenset:
                    self.graph.edge(origem,destino, label = transicao)
                elif type(origem) is frozenset and type(destino) is not frozenset:
                    m = []
                    for i in origem:
                        m.append(i)
                    orig = ''.join(sorted(m))
                    self.graph.edge(orig,destino, label = transicao)
                elif type(origem) is not frozenset and type(destino) is frozenset:
                    m = []
                    for i in destino:
                        m.append(i)
                    dest = ''.join(sorted(m))
                    self.graph.edge(origem,dest, label = transicao)
                else:
                    m = []
                    for i in destino:
                        m.append(i)
                    dest = ''.join(sorted(m))
                    n = []
                    for i in origem:
                        n.append(i)
                    orig = ''.join(sorted(n))
                    self.graph.edge(orig,dest, label = transicao)


                    

def NFAtoDFA(N):
    
    q0 = frozenset([N.q0])
    Q = set([q0])
    Qcopia = Q.copy()
    auto = {}
    F = []
    Alfabeto =  N.alphabet()

    n = len(Qcopia)
    while n > 0: 
        qSet = Qcopia.pop()
        auto[qSet] = {}
        for a in Alfabeto:
            print(a,qSet,[N.autoEst(q,a) for q in qSet],Qcopia)
            if len(qSet) == 0 and len(Qcopia) == 1:
                proximos = frozenset()
            else:
                proximos = reduce(lambda x,y: x|y, [N.autoEst(q,a) for q in qSet])
                proximos = frozenset(proximos)
            auto[qSet][a] = proximos
            if not proximos in Q: 
                Q.add(proximos)
                Qcopia.add(proximos)
            n = len(Qcopia)
    for qSet in Q: 
        if len(qSet & N.F) > 0: 
            F.append(qSet)
    convertido = DFA(auto, q0, F)
    
    return convertido

entrada2 ={'0':{'a':set(['3'])},'3':{'b':set(['1','2']),'c':set(['1','2'])},'2':{'a':set(['3'])}}
entrada3 = {'0':{'a':set(['1','4']),'b':set(['1','5'])},'3':{'a':set(['1'])},'2':{'b':set(['1'])},'4':{'a':set(['2','4']),'b':set(['1'])},'5':{'a':set(['1']),'b':set(['3','5'])}}
entrada1 = {'0':{'a':set(['3']),'b':set(['3','4'])},'3':{'a':set(['3','2']),'b':set(['3','2','4'])},'2':{'b':set(['4'])},'4':{'b':set(['1','5','6'])},'5':{'a':set(['6']),'b':set(['6'])},'6':{'a':set(['1','6']),'b':set(['1','6'])}}

N = NFA(entrada1,'0',['1','5','6'])
M1 = NFAtoDFA(N)
M1.generateFormat()
print(M1.graph.source)

N = NFA(entrada2,'0',['1','2'])
M2 = NFAtoDFA(N)
M2.generateFormat()
print(M2.graph.source)

N = NFA(entrada3,'0',['1'])
M3 = NFAtoDFA(N)
M3.generateFormat()
print(M3.graph.source)