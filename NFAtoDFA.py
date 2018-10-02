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
        self.finaismin = []
        self.tabela = {}
        self.q0 = estadoInicial
        self.F = estadosFinais
        self.graph = Digraph(comment='DFA')
        self.alfa = alfabeto
        self.min = {}
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
                    self.finaismin.append(str(i))
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
                            self.min[str(novo_estado[k])] = {transicao:str(novo_estado[destino])}
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
entrada4 = {'0':{'a':set(['3']),'b':set(['5']),'c':set(['6'])},'2':{'b':set(['5']),'c':set(['6'])},'3':{'a':set(['1','2','3','4','5','6']),'b':set(['5']),'c':set(['6'])},'4':{'c':set(['6'])},'5':{'b':set(['1','4','6','5']),'c':set(['6'])},'6':{'c':set(['1','6'])}}
N = NFA(entrada1,'0',['1','5','6'])
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


#N = NFA(entrada3,'0',['1'])
#M3 = NFAtoDFA(N)
#M3.generateFormat()
#print(M3.graph.source)
#M3.minimiza()
#print(M3.minimizado.source)

#N = NFA(entrada4,'0',['0','1','2','3','4','5','6'])
#M4 = NFAtoDFA(N)
#M4.generateFormat()
#print(M4.graph.source)
#M4.minimiza()
#print(M4.minimizado.source)

######### Questão 3 ##########
class Automata():
    def __init__(self,expr):
        expression = ''.join(expr)
        self.start = str(0)
        self.final = str(1)
        self.new = 2
        self.nfa = {str(0):{expression:str(1)}}
        self.nfaSemEpsilon ={str(0):{expression:str(1)}}
        self.graph = Digraph(comment='E-AFN')
        self.graphe = Digraph(comment='AFN')
        self.estadosFinais = []
    
    def union(self,r1,r2):
        trans1 = ''.join(r1)
        trans2 = ''.join(r2)
        orig = trans1+'+'+trans2
        orig2 = '('+orig+')'
        for origem, c in self.nfa.items():
            for transicao,destino in self.nfa[origem].items():
                if transicao == orig or transicao == orig2:
                    aux_o = origem
                    aux_d = destino
                    aux_t = transicao
        del self.nfa[aux_o][aux_t]
        self.nfa[aux_o][trans1] = aux_d
        self.nfa[aux_o][trans2] = aux_d
                    
    
    def concat(self,r1,r2):
        trans1 = ''.join(r1)
        trans2 = ''.join(r2)
        n = len(trans2)
        orig1 = trans1+trans2
        if '+' in trans2 and n<=6:
            trans2 = '('+trans2+')'
        orig = trans1+trans2
        for origem,c  in self.nfa.items():
            for transicao,destino in self.nfa[origem].items():
                if transicao == orig or transicao == orig1:
                    aux_o = origem
                    aux_d = destino
                    aux_t = transicao
        del self.nfa[aux_o][aux_t]
        self.nfa[aux_o][trans1] = str(self.new)
        if str(self.new) not in self.nfa.keys():
            self.nfa[str(self.new)] = {trans2:aux_d}
        else: self.nfa[str(self.new)][trans2] = aux_d
        self.new = self.new+1
    
    def kleene(self,r1):
        trans = ''.join(r1)
        alt = '('+trans+')*'
        alt2 = '('+alt+')'
        for origem,c  in self.nfa.items():
            for transicao,destino in self.nfa[origem].items():
                if transicao == trans or transicao == alt or transicao == alt2:
                    aux_o = origem
                    aux_d = destino
                    aux_t = transicao
        n = len(aux_t)
        del self.nfa[aux_o][aux_t]
        if n == 2:
            aux_t = aux_t[0:1]
        else: aux_t = aux_t[1:n-2] 
        if '&' in self.nfa[aux_o].keys():
            self.nfa[aux_o]['e'] = str(self.new)
        else:
            self.nfa[aux_o]['&'] = str(self.new)
        if str(self.new) not in self.nfa.keys():
            self.nfa[str(self.new)] = {'&':aux_d}
        else: self.nfa[str(self.new)]['&'] = aux_d
        self.nfa[str(self.new)][aux_t] = str(self.new)
        self.new = self.new+1
    
    def generateFormat(self):
        for origem, caminho in self.nfa.items():
            self.graph.node(str(origem),str(origem), shape='circle')
        self.graph.node('1','1', shape='doublecircle')
        for origem, caminho in self.nfa.items():
            for transicao,destino in self.nfa[origem].items():
                if transicao == 'e':
                    self.graph.edge(str(origem),str(destino), label = '&')
                else: self.graph.edge(str(origem),str(destino), label = transicao)

    def search(self, trans, vertice, origem, paresVisitados, dictAux):
        if vertice in self.nfa:
            if trans =='&' or trans == 'e':
                for transicao,destino in self.nfa[vertice].items():
                    if (transicao,vertice) not in paresVisitados:
                        paresVisitados.append((transicao,vertice))
                        self.search(transicao, destino, origem, paresVisitados, dictAux)
            else: 
                for transicao,destino in self.nfa[vertice].items():
                    if transicao =='&' or transicao == 'e':
                        if (trans,vertice) not in paresVisitados:
                            paresVisitados.append((trans,vertice))
                            self.search(trans, destino, origem, paresVisitados, dictAux)
                dictAux[trans].append(vertice)
        elif trans !='&' and trans != 'e': 
            dictAux[trans].append(vertice)
        elif trans =='&' or trans == 'e':
            self.estadosFinais.append(origem)

    def epsilonToNoEpsilon(self):
        for origem, c in self.nfa.items():
            dictAux = {'a':[]}
            for transicao,destino in self.nfa[origem].items():
                self.search(transicao, destino, origem, [], dictAux)
            self.nfaSemEpsilon[origem] = dictAux

    def generateFormatSemE(self):
        for origem, caminho in self.nfaSemEpsilon.items():
            if origem in self.estadosFinais:
                self.graphe.node(str(origem),str(origem), shape='doublecircle')
            else: self.graphe.node(str(origem),str(origem), shape='circle')
        self.graphe.node('1','1', shape='doublecircle')
        for origem, caminho in self.nfaSemEpsilon.items():
            for transicao,destino in self.nfaSemEpsilon[origem].items():
                for node in destino:
                    if transicao == 'e':
                        self.graphe.edge(str(origem),str(node), label = '&')
                    else: self.graphe.edge(str(origem),str(node), label = transicao)
        for origem, caminho in self.nfaSemEpsilon.items():
            for transicao,destino in self.nfaSemEpsilon[origem].items():
                self.nfaSemEpsilon[origem][transicao] = set(destino)
                

def recursao(expr):
    r1 = []
    r2 = []
    previous = '&'
    testa = 1
    n = len(expr)
    if n <= 2:
        if n == 2 and '*' in expr:
            afn.kleene(expr)
        elif n == 2:
            afn.concat(expr[0],expr[1])
        return 0
    count = 0
    for i in range(0,n-1):
        if expr[i] == '(':
            count = count + 1
        elif expr[i] == ')':
            count = count - 1
        elif expr[i] == '+' and count == 0:
            r1 = expr[0:i]
            r2 = expr[i+1:n]
            testa = 0
            afn.union(r1,r2)
            recursao(r1)
            recursao(r2)
    count = 0
    for i in range(0,n):
        if expr[i] == '*':
            count = count +1
    if expr[0] == '(' and expr[n-1] == '*' and expr[n-2] == ')'and count == 1:
        #fecho de kleene
        r1 = expr[1:n-2]
        testa = 0
        print(r1)
        afn.kleene(r1)
        recursao(r1)
    count = 0
    for i in range(0,n-1):
        if testa == 1:
            if expr[i].isalpha():
                if (previous == '*' or previous.isalpha()) and count == 0:
                    r1 = expr[0:i]
                    r2 = expr[i:n]
                    afn.concat(r1,r2)
                    recursao(r1)
                    recursao(r2)
                    testa = 0
            elif expr[i] == '(':
                if previous != '&':
                    r1 = expr[0:i]
                    if expr[n-1] == '*':
                        r2 = expr[i:n]
                    elif expr[n-1] == ')':
                        r2 = expr[i+1:n-1]
                    afn.concat(r1,r2)
                    recursao(r1)
                    recursao(r2)
                    testa = 0
                count = count + 1
            elif expr[i] == ')':
                count = count - 1
                if count == 0:
                    if expr[i+1] == '*':
                        """ fecho de kleene"""
                        r1 = expr[0:i+2]
                        if i != n-2:
                            r2 = expr[i+2:n]
                        if i != n-2:
                            afn.concat(r1,r2)
                            recursao(r1)
                            recursao(r2)
                        else:
                            recursao(r1)
                        testa = 0
                    else:
                        r1 = expr[1:i]
                        recursao(r1)
                        testa = 0
            elif expr[i] == '*':
                if count == 0:
                    r1 = expr[0:i+1]
                    r2 = expr[i+1:n]
                    afn.concat(r1,r2)
                    recursao(r1)
                    recursao(r2)
                    testa = 0
            previous = expr[i]

forma = input("Digite tipo da entrada:")
if forma == "regex":
    regex1 = '(aa)*'
    regex2 = '(aaa)*'
    count = 0
    L = []
    for i in regex1:
        L.append(i)
    afn = Automata(L)
    recursao(L)
    eAFN1 = afn.nfa    
    afn.generateFormat()
    eAFN1graph = afn.graph
    afn.epsilonToNoEpsilon()
    afn.generateFormatSemE()
    AFN1 = afn.nfaSemEpsilon
    AFN1graph = afn.graphe
    N = NFA(AFN1,'0',['0','1','2'])
    M = NFAtoDFA(N)
    M.generateFormat()
    DFA1graph = M.graph
    DFA1 = M.tabela
    M.minimiza()
    DFA1min = M.min
    DFA1minGraph = M.minimizado

    L = []
    for i in regex2:
        L.append(i)
    afn = Automata(L)
    recursao(L)
    eAFN2 = afn.nfa    
    afn.generateFormat()
    eAFN2graph = afn.graph
    afn.epsilonToNoEpsilon()
    afn.generateFormatSemE()
    AFN2 = afn.nfaSemEpsilon
    AFN2graph = afn.graphe
    N = NFA(AFN2,'0',['0','1','2'])
    M = NFAtoDFA(N)
    M.generateFormat()
    DFA2graph = M.graph
    DFA2 = M.tabela
    M.minimiza()
    DFA2min = M.min
    DFA2minGraph = M.minimizado
elif forma == "eAFN":
    eAFN1 = {'0': {'&': '2'}, '2': {'&': '1', 'a': '3'}, '3': {'a': '2'}}
    eAFN2 = {'0': {'&': '2'}, '2': {'&': '1', 'a': '3'}, '3': {'a': '4'}, '4': {'a': '2'}}
    afn = Automata([])
    afn.nfa = eAFN1
    afn.generateFormat()
    eAFN1graph = afn.graph
    afn.epsilonToNoEpsilon()
    afn.generateFormatSemE()
    AFN1 = afn.nfaSemEpsilon
    AFN1graph = afn.graphe
    N = NFA(AFN1,'0',['0','1','2'])
    M = NFAtoDFA(N)
    M.generateFormat()
    DFA1graph = M.graph
    DFA1 = M.tabela
    M.minimiza()
    DFA1min = M.min
    DFA1minGraph = M.minimizado

    afn = Automata([])
    afn.nfa = eAFN2
    afn.generateFormat()
    eAFN2graph = afn.graph
    afn.epsilonToNoEpsilon()
    afn.generateFormatSemE()
    AFN2 = afn.nfaSemEpsilon
    AFN2graph = afn.graphe
    N = NFA(AFN2,'0',['0','1','2'])
    M = NFAtoDFA(N)
    M.generateFormat()
    DFA2graph = M.graph
    DFA2 = M.tabela
    M.minimiza()
    DFA2min = M.min
    DFA2minGraph = M.minimizado
elif forma == "AFN":
    AFN1 = {'0': {'a': {'3'}}, '2': {'a': {'3'}}, '3': {'a': {'2', '1'}}}
    N = NFA(AFN1,'0',['0','1','2'])
    M = NFAtoDFA(N)
    M.generateFormat()
    DFA1graph = M.graph
    DFA1 = M.tabela
    M.minimiza()
    DFA1min = M.min
    DFA1minGraph = M.minimizado

    AFN2 = {'0': {'a': {'3'}}, '2': {'a': {'3'}}, '3': {'a': {'4'}}, '4': {'a': {'2', '1'}}}
    N = NFA(AFN2,'0',['0','1','2'])
    M = NFAtoDFA(N)
    M.generateFormat()
    DFA2graph = M.graph
    DFA2 = M.tabela
    M.minimiza()
    DFA2min = M.min
    DFA2minGraph = M.minimizado
    print(DFA2min)
    print(DFA1min)
elif forma == "AFD":
    DFA1min = {'0': {'a': '1'}, '1': {'a': '0'}}
    DFA2min = {'0': {'a': '2'}, '1': {'a': '0'}, '2': {'a': '1'}}

def complementDFA(DFA, finais):
    DFAcompl = Digraph(comment='Complement')
    for origem, caminho in DFA.items():
        if origem not in finais:
            DFAcompl.node(origem,origem, shape='doublecircle')
        else: DFAcompl.node(origem,origem, shape='circle')
    for origem, caminho in DFA.items():
        for transicao,destino in DFA[origem].items():
            DFAcompl.edge(str(origem),str(destino), label = transicao)
    print(DFAcompl.source)

def unionDFA(DFA1,DFA2,finais1,finais2):
    n = len(DFA1)
    DFA2alt = {}
    for origem in DFA2.keys():
        DFA2alt[str(int(origem)+n)] = DFA2[origem]
    for origem, caminho in DFA2alt.items():
        for transicao,destino in DFA2alt[origem].items():
            DFA2alt[origem][transicao] = str(int(destino)+n)
    union ={'q0':{'&':'0','e':str(n)}}
    for i in range(0,len(finais2)):
        finais2[i] = str(int(finais2[i])+n)
    union.update(DFA1)
    union.update(DFA2alt)
    afn = Automata([])
    afn.start = str('q0')
    afn.estadosFinais = finais1 + finais2 + ['q0']
    afn.nfa = union
    afn.generateFormat()
    afn.epsilonToNoEpsilon()
    afn.generateFormatSemE()
    AFN = afn.nfaSemEpsilon
    print(AFN)
    N = NFA(AFN,'q0',afn.estadosFinais)
    M = NFAtoDFA(N)
    M.generateFormat()
    DFA1 = M.tabela
    M.minimiza()
    print(M.minimizado.source)

def intersectionDFA(DFA1,DFA2,finais1,finais2):
    final1 = []
    final2 =[]
    n = len(DFA1)
    for origem in DFA1.keys():
        if origem not in finais1:
            final1.append(origem)
    for origem in DFA2.keys():
        if origem not in finais2:
            final2.append(str(int(origem)+n))
    DFA2alt = {}
    for origem in DFA2.keys():
        DFA2alt[str(int(origem)+n)] = DFA2[origem]
    for origem, caminho in DFA2alt.items():
        for transicao,destino in DFA2alt[origem].items():
            DFA2alt[origem][transicao] = str(int(destino)+n)
    uni ={'q0':{'&':'0','e':str(n)}}
    uni.update(DFA1)
    uni.update(DFA2alt)
    afn = Automata([])
    afn.start = str('q0')
    afn.estadosFinais = final1 + final2
    afn.nfa = uni
    afn.generateFormat()
    afn.epsilonToNoEpsilon()
    afn.generateFormatSemE()
    AFN = afn.nfaSemEpsilon
    N = NFA(AFN,'q0',afn.estadosFinais)
    M1 = NFAtoDFA(N)
    M1.generateFormat()
    M1.minimiza()
    complementDFA(M1.min,M1.finaismin)
    
#complementDFA(DFA1min,['0'])
#complementDFA(DFA2min,['0'])
#unionDFA(DFA1min,DFA2min,['0'],['0'])
#intersectionDFA(DFA1min,DFA2min,['0'],['0'])

