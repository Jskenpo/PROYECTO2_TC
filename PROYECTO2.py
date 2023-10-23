import re
import sys
from itertools import product
import numpy as np
import graphviz
import time

class CFG:    
    # Variables - Símbolos No Terminales
    _V = []
    # Alfabeto - Símbolos Terminales
    _T = []
    # Símbolo de Inicio
    _S = None
    # Producciones
    _P = []
    # Variables Aceptadas - Símbolos No Terminales - Expresión Regular
    _V_set = '[A-Z](_[0-9]*)?|(,[A-Z](_[0-9]*)?)*'
    # Alfabeto Aceptado - Símbolos Terminales - Expresión Regular
    _T_set = '.*(,.*)*'
    # Símbolo de Inicio Aceptado - Expresión Regular
    _S_set = '[A-Z](_[0-9]*)?'
    # Producciones Aceptadas - Expresión Regular
    _P_set = '([A-Z](_[0-9]*)?->.*(|.*)*(,[A-Z](_[0-9]*)?->.*(|.*)*)*)'

    #Carga una CFG desde un archivo de texto
    def cargarDesdeArchivo(self, archivoTexto):
        with open(archivoTexto) as f:
            lineas = f.readlines()
        g = ''.join([re.sub(" |\n|\t", "", x) for x in lineas])
        if not re.search('V:' + self._V_set + 'T:' + self._T_set + 'S:' + self._S_set + 'P:' + self._P_set, g):
            raise ImportError('Error: definición incorrecta de la gramática.')
        v = re.search('V:(.*)T:', g).group(1)
        t = re.search('T:(.*)S:', g).group(1)
        s = re.search('S:(.*)P:', g).group(1)
        p = re.search('P:(.*)', g).group(1)
        self.cargar(v, t, s, p)

    #Carga una CFG desde cadenas de texto
    def cargar(self, v, t, s, p):
        self._V = [re.escape(x) for x in re.split(',', v.replace(" ", ""))]
        self._T = [re.escape(x) for x in re.split(',', t.replace(" ", ""))]
        if [x for x in self._V if x in self._T]:
            sys.exit('Error: la intersección entre V y T no está vacía')
        s = re.escape(s.replace(" ", ""))
        if s in self._V:
            self._S = s
        else:
            sys.exit('Error: el símbolo de inicio no está en V')
        p = p.replace(" ", "")
        self._P = self._parsProductions(p)

    #Analiza las producciones y las almacena en una estructura de datos
    def _parsProductions(self, p):
        P = {}
        v = []
        self.simbolos = self._V + self._T
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

    #Analiza una regla y la almacena en una estructura de datos
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


    #Crea una copia de la CFG actual
    def __copy__(self):
        return CFG().crear(self._V, self._T, self._S, self._P)

    #Método para crear una CFG
    def crear(self, v, t, s, p):
        nuevaCFG = CFG()
        nuevaCFG._V = v
        nuevaCFG._T = t
        nuevaCFG._S = s
        nuevaCFG._P = p
        return nuevaCFG

    #Método para convertir la CFG en una cadena de texto legible
    def __str__(self, order=False):
        _str = 'V: ' + ', '.join(self._V) + '\n'
        _str += 'T: ' + ', '.join(self._T) + '\n'
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
    #Verificar si una gramática está en una forma normal
    def _FN(self, CFG):
        pass

    #Convertir una gramática a una forma normal
    def _convertirFN(self, CFG):
        pass

    #Cargar una gramática CFG en la instancia de la clase
    def _cargarCFG(self, cfg):
        self._V = [x for x in cfg._V]
        self._T = [x for x in cfg._T]
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
                
        return CFG().crear(self._V, self._T, self._S, self._P)

    #Eliminar producciones epsilon de una gramática CFG
    def _produccionesEpsilon(self):
        if re.escape('epsilon') not in self._T:
            return
        self._T = [x for x in self._T if x is not re.escape('epsilon')]
        _P = {}
        for v in self._V:
            if v not in _P.keys():
                _P[v] = []
            for p in self._P[v]:
                if len(p) == 1 and p[0] == re.escape('epsilon'):
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
            if len(p) > 1 or p[0] in self._T:
                T = [x for x in [p] if x not in T] + T
            elif p[0] != parent:
                T = [x for x in self._encontrarTerminales(parent, p[0]) if x not in T] + T
        return T

    #Reducir la gramática eliminando símbolos no alcanzables
    def _reducirCFG(self):
        W = {}
        W[0] = self._actW(self._T)
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
                if [True for x in range(len(_p))] == [x in V + self._T for n, x in _p.items()]:
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
        self._T = [x for x in self._T if x in Y[j]]

    #Propagar producciones de una gramática CFG y encontrar símbolos alcanzables
    def _propagarProduccion(self, Y, _prev=None):
        _y = [x for x in Y]
        y = [x for x in Y if x not in self._T]
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
    _simbolos_generados = {}

    #Verifica si la gramática está en Forma Normal de Chomsky
    def FNC(self, cfg):
        if re.escape('epsilon') in cfg._T:
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
        self._replaceMixedTerminals()
        self._dividirSecuenciasNT()
        self._produccionesUnarias()
        
        return CFG().crear(self._V, self._T, self._S, self._P)

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
    def _replaceMixedTerminals(self):
        _P = {}
        _wrongPs = {}
        for v, Ps in self._P.items():
            _P[v] = []
            for p in Ps:
                if len(p) > 1 and len([x for x in p.values() if x in self._T]) > 0:
                    if v not in _wrongPs.keys():
                        _wrongPs[v] = []
                    _wrongPs[v].append(p)
                else:
                    _P[v].append(p)
        _conv = {}
        for v, Ps in _wrongPs.items():
            for p in Ps:
                for s in list(set([y for y in [x for x in p.values() if x in self._T] if y not in _conv.keys()])):
                    _v = self._crearVariable(v[0])
                    self._V.append(_v)
                    _conv[s] = _v
                    _P[_v] = [{0: s}]
                _p = {}
                for j, s in p.items():
                    if s in self._T:
                        _p[j] = _conv[s]
                    else:
                        _p[j] = s
                _P[v].append(_p)
        self._P = _P


class CYK:
    def __init__(self, grammar):
        self.grammar = grammar
        self._V = self.grammar._V
        self._T = self.grammar._T
        self._S = self.grammar._S
        self._P = self.grammar._P
        self._preprocesar_gramatica()
    
    def _preprocesar_gramatica(self):
        for v in self.grammar._P:
            for prod in self.grammar._P[v]:
                for i, s in prod.items():
                    if s in self.grammar._T:
                        # Quitar escapes
                        prod[i] = s.replace('\\', '')
        self.grammar._T = [s.replace('\\', '') for s in self.grammar._T]

    def validate(self, cadena):
        cadena = cadena.split(' ')
        n = len(cadena)
        P = self._P
        V = self._V
        S = self._S
        T = self._T
        M = np.empty((n, n), dtype=list)

        ##Llenar diagonal dentro de la matriz
        for i in range(n):
            for v in V:
                if {0: cadena[i]} in P[v]:
                    M[i, i] = [v] if M[i, i] is None else M[i, i] + [v]

        ##Limprimir diagonal en la tabla 
        print('Tabla CYK')
        for i in range(n):
            for j in range(n):
                print(M[i, j], end='\t')
            print()

        ##Llenar el resto de la matriz
        for l in range(1, n):
            for i in range(n - l):
                j = i + l
                for k in range(i, j):
                    for v in V:
                        for p in P[v]:
                            if len(p) == 2:
                                if M[i, k] is None or M[k + 1, j] is None:
                                    continue
                                if p[0] in M[i, k] and p[1] in M[k + 1, j]:
                                    M[i, j] = [v] if M[i, j] is None else M[i, j] + [v]

        ## imprimir tabla
        print('\nTabla CYK')
        for i in range(n):
            for j in range(n):
                print(M[i, j], end='\t')
            print()

        ##Validar si la cadena es aceptada
        if  M[0, n - 1] is None:
            result = False
        elif S in M[0, n - 1] :
            result = True
        else:
            result = False

        ##Imprimir resultado
        if result:
            print(f"\nLa cadena pertenece a la gramática y es acepta.")
        else:
            print(f"\nLa cadena no pertenece a la gramática y no es acepta.")

        return result

    def generate_parse_tree(self, cadena):
        cadena = cadena.split(' ')
        n = len(cadena)
        P = self._P
        V = self._V
        S = self._S
        T = self._T
        M = np.empty((n, n), dtype=list)

        ##Llenar diagonal dentro de la matriz
        for i in range(n):
            for v in V:
                if {0: cadena[i]} in P[v]:
                    M[i, i] = [v] if M[i, i] is None else M[i, i] + [v]

        ##Llenar el resto de la matriz
        for l in range(1, n):
            for i in range(n - l):
                j = i + l
                for k in range(i, j):
                    for v in V:
                        for p in P[v]:
                            if len(p) == 2:
                                if M[i, k] is None or M[k + 1, j] is None:
                                    continue
                                if p[0] in M[i, k] and p[1] in M[k + 1, j]:
                                    M[i, j] = [v] if M[i, j] is None else M[i, j] + [v]

        # Asumiendo que la cadena es aceptada, genera el parse tree
        if M[0, n - 1] is not None and S in M[0, n - 1]:
            dot = graphviz.Digraph(format='png')

            def generate_tree(i, j, v):
                if i == j:
                    dot.node(f'{i},{j}', label=f'{cadena[i]}')
                else:
                    dot.node(f'{i},{j}', label=v)
                    for production in P[v]:
                        if len(production) == 2:
                            left_v, right_v = list(production.values())
                            for k in range(i, j):
                                if M[i, k] is not None and M[k + 1, j] is not None:
                                    if left_v in M[i, k] and right_v in M[k + 1, j]:
                                        generate_tree(i, k, left_v)
                                        generate_tree(k + 1, j, right_v)
                                        dot.edge(f'{i},{j}', f'{i},{k}')
                                        dot.edge(f'{i},{j}', f'{k + 1},{j}')
            
            generate_tree(0, n - 1, S)
            dot.render('parse_tree', view=True)
    

if __name__ == "__main__":
    print("Algoritmo de simplificación de gramáticas y algoritmo CYK\n")
    G = CFG()
    expresion = 'id + ( id ) * id'
    #expresion = 'the cat drinks the beer'
    #expresion = 'she eats'

    G.cargarDesdeArchivo('gramatica.txt')
    #G.cargarDesdeArchivo('gramatica2.txt')
    #G.cargarDesdeArchivo('gramatica3.txt')
    resultado = ChomskyFN().FNC(G)
    print("Gramática original:")
    print(G)

    ## Forma normal de chomsky
    print("\nNueva Gramática en Forma Normal de Chomsky:")
    G = CFG()
    G.cargarDesdeArchivo('gramatica.txt')
    #G.cargarDesdeArchivo('gramatica2.txt')
    #G.cargarDesdeArchivo('gramatica3.txt')
    resultado = ChomskyFN().convertToNF(G)
    print(resultado)

    print("\nValidación de Cadenas:")
    inicio = time.time()
    c = CYK(resultado)
    print(c.validate(expresion))
    c.generate_parse_tree(expresion)
    fin = time.time()

    print("\nTiempo de ejecución: ", fin - inicio, "segundos")
