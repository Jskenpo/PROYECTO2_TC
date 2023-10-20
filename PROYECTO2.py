'''
PROYECTO 2 - TEORÍA DE LA COMPUTACIÓN
INTEGRANTES:
JOSE SANTISTEBAN
SEBASTIÁN SOLORZANO
MANUEL RODAS 

DESCRIPCIÓN:
El programa toma como entrada un archivo con una CFG y el proposito es convertira a la forma normal de chomsky y validar la gramática

'''


import re
import sys
from itertools import product

class CFG:
    """ Context Free Gramamr Class """
    # Variables - Non Terminal Symbols
    _V = []
    # Alphabet - Terminal Symbols
    _SIGMA = []
    # Start Symbol
    _S = None
    # Productions
    _P = []
    # Accepted Variables - Non Terminal Symbols - RegExp
    _V_set = '[A-Z](_[0-9]*)?(,[A-Z](_[0-9]*)?)*'
    # Accepted Alphabet - Terminal Symbols - RegExp
    _SIGMA_set = '.*(,.*)*'
    # Accepted Start Symbol - RegExp
    _S_set = '[A-Z](_[0-9]*)?'
    # Accepted Productions - RegExp
    _P_set = '([A-Z](_[0-9]*)?->.*(|.*)*(,[A-Z](_[0-9]*)?->.*(|.*)*)*)'

    def loadFromFile(self, txtFile):
        """ Costructor From File """
        with open(txtFile) as f:
            lines = f.readlines()
        g = ''.join([re.sub(" |\n|\t", "", x) for x in lines])
        if not re.search('V:' + self._V_set + 'SIGMA:' + self._SIGMA_set + 'S:' + self._S_set + 'P:' + self._P_set, g):
            raise ImportError('Error : grammar bad definition, define your grammar as :'
                              '\nV:[V|V_0],...\nSIGMA:[s|#],...\nS:s0\nP:V1->s1V|#,V2->s1|s2|...')
        v = re.search('V:(.*)SIGMA:', g).group(1)
        sigma = re.search('SIGMA:(.*)S:', g).group(1)
        s = re.search('S:(.*)P:', g).group(1)
        p = re.search('P:(.*)', g).group(1)
        self.load(v, sigma, s, p)

    def load(self, v, sigma, s, p):
        """ Costructor From Strings """
        self._V = [re.escape(x) for x in re.split(',', v.replace(" ", ""))]
        self._SIGMA = [re.escape(x) for x in re.split(',', sigma.replace(" ", ""))]
        if [x for x in self._V if x in self._SIGMA]:
            sys.exit('Error : V intersection SIGMA is not empty')
        s = re.escape(s.replace(" ", ""))
        if s in self._V:
            self._S = s
        else:
            sys.exit('Error : start symbol is not in V')
        p = p.replace(" ", "")
        self._P = self._parsProductions(p)

    def _parsProductions(self, p):
        """ Productions Builder """
        P = {}
        v = []
        self.symbols = self._V + self._SIGMA
        rows = re.split(',', p)
        for row in rows:
            item = re.split('->', row)
            left = re.escape(item[0])
            if (left in self._V):
                v.append(left)

                if left not in P:
                    P[left] = []
                rules = re.split('\|', item[1])
                for rule in rules:
                    P[left].append(self._computeRule(rule))
            else:
                raise ImportError('Rigth simbol in production ' + row + ' is not in V')
                
        for symbol in self._V:
            if symbol not in v:
            # Añadir símbolos no usados con una producción vacía
                P[symbol] = [{}]
        return P

    def _computeRule(self, rule):
        """ Single Rule Builder"""
        _rule = rule
        rules = {}
        i = 0
        while len(_rule) > 0:
            r = re.search('|'.join(self.symbols), rule)
            if r.start() == 0:
                rules[i] = re.escape(_rule[0:r.end()])
                _rule = _rule[r.end():]
                i += 1
            else:
                raise ImportError('Error : undefined symbol find in production : ' + _rule)
        return rules

    def __copy__(self):
        """ Copy Costructor """
        return CFG().create(self._V, self._SIGMA, self._S, self._P)

    def create(self, v, sigma, s, p):
        """ Static Costructor """
        newCFG = CFG()
        newCFG._V = v
        newCFG._SIGMA = sigma
        newCFG._S = s
        newCFG._P = p
        return newCFG

    def __str__(self, order=False):
        _str = 'V: ' + ', '.join(self._V) + '\n'
        _str += 'SIGMA: ' + ', '.join(self._SIGMA) + '\n'
        _str += 'S: ' + self._S + '\n'
        _str += 'P:'
        if order:
            V = [x for x in order if x in self._V] + [x for x in self._V if x not in order]
        else:
            V = self._V
        for v in V:
            _str += '\n\t' + v + ' ->'
            _PS = []
            for p in self._P[v]:
                _p = ''
                for i, s in p.items():
                    _p += ' ' + s
                _PS.append(_p)
            _str += ' |'.join(_PS)
        return _str.replace('\\', '')

class GenericNF(object):
    """ Generic Normal Form Class """

    def isInNF(self, CFG):
        pass

    def convertToNF(self, CFG):
        pass

    def _loadCFG(self, cfg):
        self._V = [x for x in cfg._V]
        self._SIGMA = [x for x in cfg._SIGMA]
        self._S = cfg._S
        self._P = {}
        for v, p in cfg._P.items():
            self._P[v] = []
            for el in p:
                _p = {}
                for i, s in el.items():
                    _p[i] = s
                self._P[v].append(_p)

    def simplifyCFG(self, cfg):
        """ Base Normal Form : epsilon-free Grammar """
        self._loadCFG(cfg)
        self._removeNullProductins()
        self._removeUnitProductins()
        self._reduceCFG()
        
        
        return CFG().create(self._V, self._SIGMA, self._S, self._P)

    def _removeNullProductins(self):
        if re.escape('#') not in self._SIGMA:
            return
        self._SIGMA = [x for x in self._SIGMA if x is not re.escape('#')]
        _P = {}
        for v in self._V:
            if v not in _P.keys():
                _P[v] = []
            for p in self._P[v]:
                if len(p) == 1 and p[0] == re.escape('#'):
                    newPs = self._createProductions(v)
                    for _v, _p in newPs.items():
                        if _v not in _P.keys():
                            _P[_v] = []
                        _P[_v] = [x for x in _p if x not in _P[_v]] + _P[_v]
                else:
                    _P[v] = [x for x in [p] if x not in _P[v]] + _P[v]
        self._P = _P

    def _createProductions(self, s):
        _P = {}
        for v in self._V:
            for p in self._P[v]:
                if s in p.values():
                    if len(p.values()) > 1:
                        # generate all possible combination
                        i = list(p.values()).count(s)
                        cases = [[x for x in l] for l in list(product([True, False], repeat=i))]
                        # [Treu]*i means that all ss do not change eg. s=B, V->aBa remain aBa
                        cases = [x for x in cases if x != [True] * i]
                        for case in cases:
                            k = 0  # production length
                            _i = 0  # number of s appeared
                            c = {}
                            for key, val in p.items():
                                if val != s:
                                    c[k] = val
                                    k += 1
                                elif case[_i]:
                                    c[k] = val
                                    k += 1
                                    _i += 1
                                else:
                                    _i += 1
                            if v not in _P.keys():
                                _P[v] = []
                            _P[v] = [x for x in [c] if x not in _P[v] and x != {}] + _P[v]
                    else:
                        # this mean that v -> p is equl to v -> #
                        newPs = self._createProductions(v)
                        for _v, _p in newPs.items():
                            if _v not in _P.keys():
                                _P[_v] = []
                            _P[_v] = [x for x in _p if x not in _P[_v]] + _P[_v]
        return _P

    def _removeUnitProductins(self):
        P = {}
        for v in self._V:
            P[v] = []
            for p in self._P[v]:
                if len(p) == 1 and p[0] in self._V:
                    newPs = self._findTerminals(v, p[0])
                    P[v] = [x for x in newPs if x not in P[v]] + P[v]
                else:
                    P[v] = [x for x in [p] if x not in P[v]] + P[v]
        self._P = P
        self._reduceCFG()

    def _findTerminals(self, parent, son):
        T = []
        for p in self._P[son]:
            if len(p) > 1 or p[0] in self._SIGMA:
                T = [x for x in [p] if x not in T] + T
            elif p[0] != parent:
                T = [x for x in self._findTerminals(parent, p[0]) if x not in T] + T
        return T

    def _reduceCFG(self):
        W = {}
        W[0] = self._updateW(self._SIGMA)
        i = 1
        W[i] = self._updateW(W[i - 1], W[i - 1])
        while (W[i] != W[i - 1]):
            i += 1
            W[i] = self._updateW(W[i - 1], W[i - 1])
        V = W[i]
        _P = {}
        for v in V:
            _P[v] = []
            for _p in self._P[v]:
                if [True for x in range(len(_p))] == [x in V + self._SIGMA for n, x in _p.items()]:
                    _P[v].append(_p)
        self._P = _P
        Y = {}
        Y[0] = [self._S]
        j = 1
        Y[1] = self._propagateProduction(Y[0])
        while (Y[j] != Y[j - 1]):
            j += 1
            Y[j] = self._propagateProduction(Y[j - 1], Y[j - 2])
        self._V = [x for x in V if x in Y[j]]
        self._SIGMA = [x for x in self._SIGMA if x in Y[j]]

    def _propagateProduction(self, Y, _prev=None):
        _y = [x for x in Y]
        y = [x for x in Y if x not in self._SIGMA]
        if _prev is not None:
            y = [x for x in y if x not in _prev]
        for v in y:
            for p in self._P[v]:
                for n, s in p.items():
                    if s not in Y:
                        _y.append(s)
        return _y

    def _updateW(self, SET, _prev=None):
        if _prev is not None:
            W = [x for x in _prev]
        else:
            W = []
        for v in self._P:
            for p in self._P[v]:
                for n, _v in p.items():
                    if _v in SET and v not in W:
                        W.append(v)
        return W

class ChomskyNF(GenericNF):
    """ Chomsky Normal Form Class """
    _gen_symbols = {}

    def isInNF(self, cfg):
        """ X -> a | YZ  """
        if re.escape('#') in cfg._SIGMA:
            return False
        else:
            for v, PS in cfg._P.items():
                for p in PS:
                    if len(p) > 2:
                        return False
                    elif len(p) == 2 and (p[0] not in cfg._V or p[1] not in cfg._V):
                        return False
                    elif len(p) == 1 and p[0] in cfg._V:
                        return False
        return True

    def convertToNF(self, cfg):
        self._loadCFG(cfg)
        self._reduceCFG()
        self._removeNullProductins()
        self._removeUnitProductins()
        self._splitNonTerminalSequences()
        self.replaceTerminals()
        
        
        
        return CFG().create(self._V, self._SIGMA, self._S, self._P)


    def _createVariable(self, S):
        i = 0
        while (S + '\\_' + str(i) in self._V):
            i += 1
        return S + '\\_' + str(i)

    def _splitNonTerminalSequences(self):
        _P = {}
        _wrongPs = {}
        for v, Ps in self._P.items():
            _P[v] = []
            for p in Ps:
                if len(p) > 2 and len([x for x in p.values() if x in self._V]) > 0:
                    if v not in _wrongPs.keys():
                        _wrongPs[v] = []
                    _wrongPs[v].append(p)
                else:
                    _P[v].append(p)
        for v, Ps in _wrongPs.items():
            if v not in _P.keys():
                _P[_v[j]] = []
            for p in Ps:
                n = len(p)
                _v = {0: v}
                for j in range(1, n):
                    if j != n - 1:
                        _v[j] = self._createVariable('X')
                        self._V.append(_v[j])
                    else:
                        _v[j] = p[j]
                    if _v[j] not in _P.keys():
                        _P[_v[j]] = []
                    _P[_v[j - 1]].append({0: p[j - 1], 1: _v[j]})
        self._P = _P
    
    def replaceTerminals(self):
        _P = {}
        nuevasProds = {}
        for v, Ps in self._P.items():
            _P[v] = []
            for p in Ps:
                _p = {}
                for j, s in p.items():
                    if s in self._SIGMA:
                        if s not in nuevasProds.keys():
                            nuevasProds[s] = self._createVariable(s)
                            self._V.append(nuevasProds[s])
                            _P[nuevasProds[s]] = [{0: s}]
                        _p[j] = nuevasProds[s]
                    else:
                        _p[j] = s
                _P[v].append(_p)
        self._P = _P

if __name__ == "__main__":
    print("Chomsky Normal Form")
    G = CFG()

    print('\nTest : check normal form (gramatica.txt)')
    G.loadFromFile('gramatica.txt')
    result = ChomskyNF().isInNF(G)
    print(G)

    if result:
        print('\ngrammar is in Chomsky normal form')
    else:
        print('\ngrammar is not in Chomsky normal form\n')
        g = ChomskyNF().convertToNF(G)
        print(g)

        if ChomskyNF().isInNF(g):
            print('\ngrammar is now in Chomsky normal form')
        else:
            print('\ngrammar is still not in Chomsky normal form')
