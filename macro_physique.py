import numpy as np
import matplotlib.pyplot as plt

## QBC


def tirage(memes,fitness):
    """tirage aléatoire de meme selon une probabilité proportionnelle à fitness"""
    fit = [fitness[j] for j in memes]
    F2 = sum(fit)
    proba = np.random.rand()*F2
    seuil = 0
    for i in range(len(fit)):
        seuil += fit[i]
        if seuil > proba:
            return memes[i]
    return "problématic"


def operation_macro(Nff,Memoire,Fitness,alpha,mu,user,F):
    """macroscopic QBC model"""
    Nf = Nff.copy()
    M = Memoire.copy()
    m_proba = np.random.random()

    #Création d'1 meme
    if m_proba < mu:

        #perte de memoire
        for e in M[user]:
            Nf[Fitness[e]-1]-=1
        if len(M[user])==alpha:
            M[user] = M[user][1:]
        #meme crée
        M[user].append(len(Fitness))
        f=np.random.randint(1,F+1)
        Fitness.append(f)
        Nf[f-1]+=alpha


    #Transmission d'1 meme
    else:
        #graphe de degré 1
        receiver = user-1

        meme = tirage(M[user],Fitness)

        #Si le meme n'est pas dans la memoire de receiver
        if not meme in M[receiver]:

            #perte de memoire
            for e in M[receiver]:
                Nf[Fitness[e]-1]-=1
            if len(M[receiver])==alpha:
                M[receiver] = M[receiver][1:]

            #meme transmis
            M[receiver].append(meme)
            Nf[Fitness[meme]-1]+=alpha

    return Nf,M,Fitness



def init_macro(Nu,alpha,mu,F):
    """Génère un système plein au sens où tous les individus ont leur mémoire remplie"""

    #Les fitness de chaque meme
    Fitness = []

    #Le tableau mémoire, M[i][j] est le meme à la place j chez l'individu i
    M = [[] for i in range(Nu)]


    while sum([len(M[i]) for i in range(len(M))]) < alpha*Nu :

        user = np.random.randint(0,Nu)

        #Si l'user a une mémoire vide : il crée
        if M[user]==[]:

            M[user].append(len(Fitness))
            Fitness.append(np.random.randint(1,F+1))

        #Sinon on  procéde normalement
        else:
            Nf,M,Fitness=operation_macro([0]*F,M,Fitness,alpha,mu,user,F)

    return M,Fitness


def simulation_macro(Nu,alpha,mu,temps_max,F):
    """simulation du modele QBC macroscopique"""

    #initialisation
    M,Fitness=init_macro(Nu,alpha,mu,F)


    #Construction du Nf initial
    Nf = [0]*F
    for m in M:
        for j in range(len(m)):
            Nf[Fitness[m[j]]-1]+=j+1


    NfTime=[Nf.copy()]
    #Debut de simulation
    time=0
    while time < temps_max:

        #Choix de l'user
        user = np.random.randint(0,Nu)
        #Opération qu'il réalise
        Nf,M,Fitness=operation_macro(Nf,M,Fitness,alpha,mu,user,F)

        NfTime.append(Nf.copy())
        #On incremente time
        time+=1

    return NfTime,M,Fitness


## tentative Eq (18)

from sympy import symbols, Eq, solve

def stationary_solution(N,mu,F=40):

    symb="N1 N2 N3 N4 N5 N6 N7 N8 N9 N10 N11 N12 N13 N14 N15 N16 N17 N18 N19 N20 N21 N22 N23 N24 N25 N26 N27 N28 N29 N30 N31 N32 N33 N34 N35 N36 N37 N38 N39 N40"
    (N1,N2,N3,N4,N5,N6,N7,N8,N9,N10,N11,N12,N13,N14,N15,N16,N17,N18,N19,N20,N21,N22,N23,N24,N25,N26,N27,N28,N29,N30,N31,N32,N33,N34,N35,N36,N37,N38,N39,N40) = symbols(symb)
    var = (N1,N2,N3,N4,N5,N6,N7,N8,N9,N10,N11,N12,N13,N14,N15,N16,N17,N18,N19,N20,N21,N22,N23,N24,N25,N26,N27,N28,N29,N30,N31,N32,N33,N34,N35,N36,N37,N38,N39,N40)
    s=0
    for j in range(F):
        s+=(j+1)*var[j]
    EQ=[]
    for i in range(F):
        EQ.append(Eq((-var[i]/N + mu/F+ (1-mu)*(i+1)*var[i]/s),0))

    print(len(EQ),EQ[0])

    solution = solve(tuple(EQ),var)
    return solution

sol = stationary_solution(N=10**3,mu=0.1)



## Simulation
plt.close('all')


timax = 10**6

Nu_ = 10**3
alpha_ = 10
F_ = 40

fitness = np.linspace(1/40,1+1/40,40)

for mu_ in [0.1,0.4,0.7,1]:
    print(mu_)
    NfTime_,Memoire,Fitness_=simulation_macro(Nu=Nu_,alpha=alpha_,mu=mu_,temps_max=timax,F=F_)

    NT = np.array([NfTime_[-1]])

    # Graphes MACRO : Density / Fitness
    nf = [0]*F_
    for f in range(F_):
        nf[f]=NfTime_[-1][f]/(Nu_*alpha_/F_)

    plt.plot(fitness,nf,label = "QBC, µ ="+str(mu_))



plt.xlim(0,1)
plt.xlabel("Fitness")
plt.ylabel("Density")
plt.title = "Analyse Macroéconomique"
plt.legend()

plt.show()


