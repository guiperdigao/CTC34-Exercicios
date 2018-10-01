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
    def __init__(self, automato, estadoInicial, estadosFinais, alfabeto):
        self.auto = automato
        self.tabela = {}
        self.q0 = estadoInicial
        self.F = estadosFinais
        self.graph = Digraph(comment='DFA')
        self.alfa = alfabeto
        self.minimizado = Digraph(comment='DFA-min')

    def autoEst(self, state, cadeia):
        # Define a função de transição estendida (retorna o estado P onde o DFA vai estar após ler a cadeia)
        for a in cadeia: 
            state = self.auto[state][a]
        return state
        
    def inLanguage(self, cadeia):
        return self.autoEst(self.q0, cadeia) in self.F

    def generateFormat(self):
        for origem, caminho in self.auto.items():
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
                    if origem not in self.tabela.keys():
                        self.tabela[origem] = {transicao:destino}
                    else: self.tabela[origem][transicao] = destino
                elif type(origem) is frozenset and type(destino) is not frozenset:
                    m = []
                    for i in origem:
                        m.append(i)
                    orig = ''.join(sorted(m))
                    self.graph.edge(orig,destino, label = transicao)
                    if orig not in self.tabela.keys():
                        self.tabela[orig] = {transicao:destino}
                    else: self.tabela[orig][transicao] = destino
                elif type(origem) is not frozenset and type(destino) is frozenset:
                    m = []
                    for i in destino:
                        m.append(i)
                    dest = ''.join(sorted(m))
                    self.graph.edge(origem,dest, label = transicao)
                    if origem not in self.tabela.keys():
                        self.tabela[origem] = {transicao:dest}
                    else: self.tabela[origem][transicao] = dest
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
                    if orig not in self.tabela.keys():
                        self.tabela[orig] = {transicao:dest}
                    else: self.tabela[orig][transicao] = dest
    
    def minimiza(self):
        W = []
        for origem, caminho in self.auto.items():
            if origem in self.F:
                if type(origem) is not frozenset:
                    W.append(origem)
                else:
                    m = []
                    for i in origem:
                        m.append(i)
                    orig = ''.join(sorted(m))
                    W.append(orig)
        finais = W
        aux = []
        P = []
        for origem, linhas in self.tabela.items():
            aux.append(origem)
        for i in aux:
            if i not in W:
                P.append(i)
        P = [set(W), set(P)]
        W = [set(W)]
        while len(W) > 0:
            A = W.pop()
            for i in self.alfa:
                X = []
                for origem, linhas in self.tabela.items():
                    for transicao,destino in self.tabela[origem].items():
                        if i == transicao:
                            if destino in A:
                                X.append(origem)
                X = [set(X)]
                n = len(P)
                for k in range(0,n):
                    Y = [P[k]]
                    intersection = []
                    difference = []
                    is_final = True
                    for y in Y[0]:
                        if y in X[0]:
                            intersection.append(y)
                        else: difference.append(y)
                        if y not in W:
                            is_final = False
                    if len(intersection) != 0 and len(difference) != 0:
                        P[k] = intersection
                        P.append(difference)
                        if is_final == True:
                            for m in range(0,len(W)):
                                if W[m] in Y:
                                    del W[m]
                            W.append(intersection)
                            W.append(difference)
                        else:
                            if len(intersection) <= len(difference):
                                W.append(intersection)
                            else: W.append(difference)
        n = len(P)
        novo_estado = {}
        for i in range(n-1,-1,-1):
            for k in P[i]:
                if k in finais:
                    self.minimizado.node(str(i),str(i), shape='doublecircle')
                else: self.minimizado.node(str(i),str(i), shape='circle')
                break 
        for i in range(0,n):
            for k in P[i]:
                for origem, linhas in self.tabela.items():
                    if k == origem:
                        novo_estado[k] = i
        for i in range(0,n):
            for k in P[i]:
                for origem, linhas in self.tabela.items():
                    for transicao,destino in self.tabela[origem].items():
                        if k == origem:
                            self.minimizado.edge(str(novo_estado[k]),str(novo_estado[destino]), label = transicao)
                break

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
    convertido = DFA(auto, q0, F, Alfabeto)
    
    return convertido

entrada2 ={'0':{'a':set(['3'])},'3':{'b':set(['1','2']),'c':set(['1','2'])},'2':{'a':set(['3'])}}
entrada3 = {'0':{'a':set(['1','4']),'b':set(['1','5'])},'3':{'a':set(['1'])},'2':{'b':set(['1'])},'4':{'a':set(['2','4']),'b':set(['1'])},'5':{'a':set(['1']),'b':set(['3','5'])}}
entrada1 = {'0':{'a':set(['3']),'b':set(['3','4'])},'3':{'a':set(['3','2']),'b':set(['3','2','4'])},'2':{'b':set(['4'])},'4':{'b':set(['1','5','6'])},'5':{'a':set(['6']),'b':set(['6'])},'6':{'a':set(['1','6']),'b':set(['1','6'])}}
entrada4 = {'0':{'a':set(['3']),'b':set(['5']),'c':set(['6'])},'2':{'b':set(['5']),'c':set(['6'])},'3':{'a':set(['1','2','3','4','5','6']),'b':set(['5'])},'4':{'c':set(['6'])},'5':{'b':set(['1','4','6']),'c':set(['6'])},'6':{'c':set(['1','6'])}}
#N = NFA(entrada1,'0',['1','5','6'])
#M1 = NFAtoDFA(N)
#M1.generateFormat()
#print(M1.graph.source)
#M1.minimiza()
#print(M1.minimizado.source)

#N = NFA(entrada2,'0',['0','1','2'])
#M2 = NFAtoDFA(N)
#M2.generateFormat()
#print(M2.graph.source)
#M2.minimiza()
#print(M2.minimizado.source)


N = NFA(entrada3,'0',['1'])
M3 = NFAtoDFA(N)
M3.generateFormat()
print(M3.graph.source)
M3.minimiza()
print(M3.minimizado.source)

#N = NFA(entrada4,'0',['0','1','2','3','4','5','6'])
#M4 = NFAtoDFA(N)
#M4.generateFormat()
#print(M4.graph.source)

