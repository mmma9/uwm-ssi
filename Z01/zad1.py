import numpy as np
import numpy
import collections

plik = open('diabetes.txt','r')

tab = []
for linia in plik:
   t = linia.split(" ")
   t = [float(x) for x in t]
   tab.append(t)

odchylenie_standardowe = np.std(tab, axis=0)

wartosci_min = np.min(tab, axis=0)
wartosci_max = np.max(tab, axis=0)

array = numpy.array(tab)

odchylenie_standardowe_systemu = np.std(array)

#wypisujemy istniejące w systemie klasy decyzyjne
print("Klasy decyzyjne w systemie:")
print("Klasa 1:")
a0 = array[:,0]
print(a0)
print("Klasa 2:")
a1 = array[:,1]
print(a1)
print("Klasa 3:")
a2 = array[:,2]
print(a2)
print("Klasa 4:")
a3 = array[:,3]
print(a3)
print("Klasa 5:")
a4 = array[:,4]
print(a4)
print("Klasa 6:")
a5 = array[:,5]
print(a5)
print("Klasa 7:")
a6 = array[:,6]
print(a6)
print("Klasa 8:")
a7 = array[:,7]
print(a7)
print("Klasa 9:")
a8 = array[:,8]
print(a8)
#wielkości klas decyzyjnych (liczby obiektów w klasach)
print("Liczba obiektów w klasach i klas w systemie wynosi:")
print(array.shape)
#minimalne i maksymalne wartości poszczególnych atrybutów(dotyczy atrybutów numerycznych),
print("Minimalne wartości poszczególnych atrybutów wynoszą:")
print(wartosci_min)
print("Maksymalne wartości poszczególnych atrybutów wynoszą:")
print(wartosci_max)
#dla każdego atrybutu wypisujemy liczbę różnych dostępnych wartości,
#dla każdego atrybutu wypisujemy listę wszystkich różnych dostępnych wartości,
print("Lista oraz liczba różnych dostępnych wartości dla atrybutu 1 to:")
print(collections.Counter(a0))
print("Lista oraz liczba różnych dostępnych wartości dla atrybutu 2 to:")
print(collections.Counter(a1))
print("Lista oraz liczba różnych dostępnych wartości dla atrybutu 3 to:")
print(collections.Counter(a2))
print("Lista oraz liczba różnych dostępnych wartości dla atrybutu 4 to:")
print(collections.Counter(a3))
print("Lista oraz liczba różnych dostępnych wartości dla atrybutu 5 to:")
print(collections.Counter(a4))
print("Lista oraz liczba różnych dostępnych wartości dla atrybutu 6 to:")
print(collections.Counter(a5))
print("Lista oraz liczba różnych dostępnych wartości dla atrybutu 7 to:")
print(collections.Counter(a6))
print("Lista oraz liczba różnych dostępnych wartości dla atrybutu 8 to:")
print(collections.Counter(a7))
print("Lista oraz liczba różnych dostępnych wartości dla atrybutu 9 to:")
print(collections.Counter(a8))
#odchylenie standardowe dla poszczególnych atrybutów w całym systemie i w klasach decyzyjnych
print("Odchylenie standardowe poszczególnych atrybutów wynosi:")
print(odchylenie_standardowe)
print("Odchylenie standardowe systemu wynosi: ")
print(odchylenie_standardowe_systemu)