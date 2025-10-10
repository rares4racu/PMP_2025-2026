import numpy as np

np.random.seed(42)
sim = 100000
redb = 0

for i in range(sim):
    urn = ['red'] * 3 + ['blue'] * 4 + ['black'] * 2
    r = np.random.randint(1,7)
    if r in [2,3,5] :
        urn.append('black')
    elif r == 6 :
        urn.append('red')
    else:
        urn.append('blue')

    ans = np.random.choice(urn)
    if ans == 'red':
        redb = redb + 1

r_estimated = redb / sim
print(f"r_estimat: {r_estimated:.4f}")

# Calculul probabilitatii de a scoate o bila rosie din urna
# In urma sunt 3R,4A,2N (R pentru rosu, A pentru albastru, N pentru negru)
# Prob ca sa adaugam o bila neagra este : 3/6 => dupa ce am dat zarul avem : 3R,4A,3N => probabilitatea sa scoti o bila rosie este: 3/10
# Prob ca sa adaugam o bila rosie este : 1/6 => dupa ce am dat zarul avem : 4R,4A,2N => probabilitatea sa scoti o bila rosie este: 4/10
# Prob ca sa adaugam o bila albastra este : 2/6 => dupa ce am dat zarul avem : 3R,5A,2N => probabilitatea sa scoti o bila rosie este: 3/10
# P(rosie) = 3/6 * 3/10 + 1/6 * 4/10 + 2/6 * 3/10 = 0,15 + 0,0666 +  0,1 = 0,3166

r_theoretical = 0.3166
print(f"r_teoretic: {r_theoretical:.4f}")
diff =  r_theoretical - r_estimated
print(f"Diferenta dintre probabilitatea teoretica si cea estimata este:: {diff:.4f}")