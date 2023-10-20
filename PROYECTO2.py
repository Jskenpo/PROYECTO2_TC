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
    # Variables - Símbolos No Terminales
    _V = []
    # Alfabeto - Símbolos Terminales
    _SIGMA = []
    # Símbolo de Inicio
    _S = None
    # Producciones
    _P = []
    # Variables Aceptadas - Símbolos No Terminales - Expresión Regular
    _V_set = '[A-Z](_[0-9]*)?(,[A-Z](_[0-9]*)?)*'
    # Alfabeto Aceptado - Símbolos Terminales - Expresión Regular
    _SIGMA_set = '.*(,.*)*'
    # Símbolo de Inicio Aceptado - Expresión Regular
    _S_set = '[A-Z](_[0-9]*)?'
    # Producciones Aceptadas - Expresión Regular
    _P_set = '([A-Z](_[0-9]*)?->.*(|.*)*(,[A-Z](_[0-9]*)?->.*(|.*)*)*)'


    """ Constructor Desde Archivo: Carga una CFG desde un archivo de texto """
    def cargarDesdeArchivo(self, archivoTexto):
        with open(archivoTexto) as f:
            lineas = f.readlines()
        g = ''.join([re.sub(" |\n|\t", "", x) for x in lineas])
        if not re.search('V:' + self._V_set + 'SIGMA:' + self._SIGMA_set + 'S:' + self._S_set + 'P:' + self._P_set, g):
            raise ImportError('Error: definición incorrecta de la gramática.')
        v = re.search('V:(.*)SIGMA:', g).group(1)
        sigma = re.search('SIGMA:(.*)S:', g).group(1)
        s = re.search('S:(.*)P:', g).group(1)
        p = re.search('P:(.*)', g).group(1)
        self.cargar(v, sigma, s, p)


    """ Constructor Desde Cadenas: Carga una CFG desde cadenas de texto """
    def cargar(self, v, sigma, s, p):
        self._V = [re.escape(x) for x in re.split(',', v.replace(" ", ""))]
        self._SIGMA = [re.escape(x) for x in re.split(',', sigma.replace(" ", ""))]
        if [x for x in self._V if x in self._SIGMA]:
            sys.exit('Error: la intersección entre V y SIGMA no está vacía')
        s = re.escape(s.replace(" ", ""))
        if s in self._V:
            self._S = s
        else:
            sys.exit('Error: el símbolo de inicio no está en V')
        p = p.replace(" ", "")
        self._P = self._parsProductions(p)


    """ Constructor de Producciones: Analiza las producciones y las almacena en una estructura de datos """
    def _parsProductions(self, p):
        P = {}
        v = []
        self.simbolos = self._V + self._SIGMA
        filas = re.split(',', p)
        for fila in filas:
            item = re.split('->', fila)
            izquierda = re.escape(item[0])
            if (izquierda in self._V):
                v.append(izquierda)

                if izquierda not in P:
                    P[izquierda] = []
                reglas = re.split('\|', item[1])
                for regla in reglas:
                    P[izquierda].append(self._calcularRegla(regla))
            else:
                raise ImportError('El símbolo derecho en la producción ' + fila + ' no está en V')
                
        for simbolo in self._V:
            if simbolo not in v:
            # Agregar símbolos no utilizados con una producción vacía
                P[simbolo] = [{}]
        return P


    """ Constructor de Regla Única: Analiza una regla y la almacena en una estructura de datos """
    def _calcularRegla(self, regla):
        _regla = regla
        reglas = {}
        i = 0
        while len(_regla) > 0:
            r = re.search('|'.join(self.simbolos), regla)
            if r.start() == 0:
                reglas[i] = re.escape(_regla[0:r.end()])
                _regla = _regla[r.end():]
                i += 1
            else:
                raise ImportError('Error: se encontró un símbolo no definido en la producción: ' + _regla)
        return reglas


    """ Constructor de Copia: Crea una copia de la CFG actual """
    def __copy__(self):
        return CFG().crear(self._V, self._SIGMA, self._S, self._P)



    """ Método para crear una CFG """
    def crear(self, v, sigma, s, p):
        nuevaCFG = CFG()
        nuevaCFG._V = v
        nuevaCFG._SIGMA = sigma
        nuevaCFG._S = s
        nuevaCFG._P = p
        return nuevaCFG


    """ Método para convertir la CFG en una cadena de texto legible """
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


class Simplificacion(object):
    """ Simplificación de gramática """
    #Verificar si una gramática está en una forma normal
    def _FN(self, CFG):
        pass

    #Convertir una gramática a una forma normal
    def _convertirFN(self, CFG):
        pass

    #Cargar una gramática CFG en la instancia de la clase
    def _cargarCFG(self, cfg):
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

    #Simplificar una gramática CFG
    def simplificarCFG(self, cfg):
        self._cargarCFG(cfg)
        self._produccionesEpsilon()
        self._produccionesUnarias()
        self._reducirCFG()
                
        return CFG().crear(self._V, self._SIGMA, self._S, self._P)

    #Eliminar producciones epsilon de una gramática CFG
    def _produccionesEpsilon(self):
        if re.escape('#') not in self._SIGMA:
            return
        self._SIGMA = [x for x in self._SIGMA if x is not re.escape('#')]
        _P = {}
        for v in self._V:
            if v not in _P.keys():
                _P[v] = []
            for p in self._P[v]:
                if len(p) == 1 and p[0] == re.escape('#'):
                    nuevasProducciones = self._crearProducciones(v)
                    for _v, _p in nuevasProducciones.items():
                        if _v not in _P.keys():
                            _P[_v] = []
                        _P[_v] = [x for x in _p if x not in _P[_v]] + _P[_v]
                else:
                    _P[v] = [x for x in [p] if x not in _P[v]] + _P[v]
        self._P = _P

    #Crear producciones al eliminar producciones epsilon
    def _crearProducciones(self, s):
        _P = {}
        for v in self._V:
            for p in self._P[v]:
                if s in p.values():
                    if len(p.values()) > 1:
                        # Generar todas las combinaciones posibles
                        i = list(p.values()).count(s)
                        cases = [[x for x in l] for l in list(product([True, False], repeat=i))]
                        cases = [x for x in cases if x != [True] * i]
                        for case in cases:
                            k = 0  # longitud de la producción
                            _i = 0  # número de 's' que aparecen
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
                        nuevasProducciones = self._crearProducciones(v)
                        for _v, _p in nuevasProducciones.items():
                            if _v not in _P.keys():
                                _P[_v] = []
                            _P[_v] = [x for x in _p if x not in _P[_v]] + _P[_v]
        return _P

    #Eliminar producciones unitarias de una gramática CFG
    def _produccionesUnarias(self):
        P = {}
        for v in self._V:
            P[v] = []
            for p in self._P[v]:
                if len(p) == 1 and p[0] in self._V:
                    nuevasProducciones = self._encontrarTerminales(v, p[0])
                    P[v] = [x for x in nuevasProducciones if x not in P[v]] + P[v]
                else:
                    P[v] = [x for x in [p] if x not in P[v]] + P[v]
        self._P = P
        self._reducirCFG()

    #Encontrar producciones terminales de una gramática CFG
    def _encontrarTerminales(self, parent, son):
        T = []
        for p in self._P[son]:
            if len(p) > 1 or p[0] in self._SIGMA:
                T = [x for x in [p] if x not in T] + T
            elif p[0] != parent:
                T = [x for x in self._encontrarTerminales(parent, p[0]) if x not in T] + T
        return T

    #Reducir la gramática eliminando símbolos no alcanzables
    def _reducirCFG(self):
        W = {}
        W[0] = self._actW(self._SIGMA)
        i = 1
        W[i] = self._actW(W[i - 1], W[i - 1])
        while (W[i] != W[i - 1]):
            i += 1
            W[i] = self._actW(W[i - 1], W[i - 1])
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
        Y[1] = self._propagarProduccion(Y[0])
        while (Y[j] != Y[j - 1]):
            j += 1
            Y[j] = self._propagarProduccion(Y[j - 1], Y[j - 2])
        self._V = [x for x in V if x in Y[j]]
        self._SIGMA = [x for x in self._SIGMA if x in Y[j]]

    #Propagar producciones de una gramática CFG y encontrar símbolos alcanzables
    def _propagarProduccion(self, Y, _prev=None):
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

    #Actualizar símbolos alcanzables de una gramática CFG
    def _actW(self, SET, _prev=None):
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


class ChomskyFN(Simplificacion):
    """ Clase de Forma Normal de Chomsky """

    _simbolos_generados = {}

    #Verifica si la gramática está en Forma Normal de Chomsky
    def FNC(self, cfg):
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

    #Convertir una gramática a Forma Normal de Chomsky
    def convertToNF(self, cfg):
        self._cargarCFG(cfg)
        self._reducirCFG()
        self._produccionesEpsilon()
        self._produccionesUnarias()
        self._dividirSecuenciasNT()
        self._remplazarTerminales()
        
        return CFG().crear(self._V, self._SIGMA, self._S, self._P)

    #Crea una nueva variable única
    def _crearVariable(self, S):
        i = 0
        while (S + '\\_' + str(i) in self._V):
            i += 1
        return S + '\\_' + str(i)

    #Divide las secuencias de símbolos no terminales
    def _dividirSecuenciasNT(self):
        _P = {}
        _produccionErronea = {}
        for v, Ps in self._P.items():
            _P[v] = []
            for p in Ps:
                if len(p) > 2 and len([x for x in p.values() if x in self._V]) > 0:
                    if v not in _produccionErronea.keys():
                        _produccionErronea[v] = []
                    _produccionErronea[v].append(p)
                else:
                    _P[v].append(p)
        for v, Ps in _produccionErronea.items():
            if v not in _P.keys():
                _P[_v[j]] = []
            for p in Ps:
                n = len(p)
                _v = {0: v}
                for j in range(1, n):
                    if j != n - 1:
                        _v[j] = self._crearVariable('X')
                        self._V.append(_v[j])
                    else:
                        _v[j] = p[j]
                    if _v[j] not in _P.keys():
                        _P[_v[j]] = []
                    _P[_v[j - 1]].append({0: p[j - 1], 1: _v[j]})
        self._P = _P
    
    #Reemplaza los símbolos terminales en las producciones
    def _remplazarTerminales(self):
        _P = {}
        nuevasProds = {}
        for v, Ps in self._P.items():
            _P[v] = []
            for p in Ps:
                _p = {}
                for j, s in p.items():
                    if s in self._SIGMA:
                        if s not in nuevasProds.keys():
                            nuevasProds[s] = self._crearVariable(s)
                            self._V.append(nuevasProds[s])
                            _P[nuevasProds[s]] = [{0: s}]
                        _p[j] = nuevasProds[s]
                    else:
                        _p[j] = s
                _P[v].append(_p)
        self._P = _P

if __name__ == "__main__":
    print("Forma Normal de Chomsky")
    G = CFG()

    G.cargarDesdeArchivo('gramatica.txt')
    resultado = ChomskyFN().FNC(G)
    print(G)

    if resultado:
        print('\nLa gramática está en Forma Normal de Chomsky')
    else:
        print('\nLa gramática no está en Forma Normal de Chomsky\n')
        g = ChomskyFN().convertToNF(G)
        print(g)

        if ChomskyFN().FNC(g):
            print('\nLa gramática ahora está en Forma Normal de Chomsky')
        else:
            print('\nLa gramática sigue sin estar en Forma Normal de Chomsky')
