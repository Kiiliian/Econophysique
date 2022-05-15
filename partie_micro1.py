import numpy as np
import matplotlib.pyplot as plt


# #Initialisation : degree = 1, Nu = 10**3, alpha = 10


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


def operation_micro(N,Memoire,Lifetime,Popularity,Fitness,Nu,alpha,mu,user,time):

    m_proba = np.random.random()

    #Création d'1 meme
    if m_proba < mu:

        #perte de memoire
        for e in Memoire[user]:
            N[e]-=1
        if len(Memoire[user])==alpha:
            Memoire[user] = Memoire[user][1:]

        #meme crée
        Memoire[user].append(len(N))
        N.append(alpha)
        Lifetime.append(-time)
        Popularity.append(1)
        Fitness.append(np.random.rand())


    #Transmission d'1 meme
    else:
        receiver = user-1

        meme = tirage(Memoire[user],Fitness)
        Popularity[meme]+=1
        #Si le meme n'est pas dans la memoire de receiver
        if not meme in Memoire[receiver]:

            #perte de memoire
            for e in Memoire[receiver]:
                N[e]-=1
            if len(Memoire[receiver])==alpha:
                Memoire[receiver] = Memoire[receiver][1:]

            #meme transmis
            Memoire[receiver].append(meme)
            N[meme]+=alpha
    return N,Memoire,Lifetime,Popularity,Fitness



def init_micro(Nu,alpha,mu):
    """Génère un système plein au sens où tous les individus ont leur mémoire remplie"""

    #La fonction N : N[meme] = N_meme
    N = []

    #Les Lifetime pour chaque meme
    Lifetime = []

    #La popularity de chaque meme
    Popularity = []

    #Le tableau mémoire, M[i][j] est le meme à la place j chez l'individu i
    Memoire = [[] for i in range(Nu)]

    #La fitness pour chaque meme
    Fitness = []

    time = 0

    while sum([len(Memoire[i]) for i in range(len(Memoire))]) < alpha*Nu :


        user = np.random.randint(0,Nu)

        #Si user a la mémoire vide : il crée un meme
        if Memoire[user]==[]:
            Memoire[user].append(len(N))
            N.append(alpha)
            Lifetime.append(-time)
            Popularity.append(1)
            Fitness.append(np.random.rand())


        #Sinon on  procéde noramlement
        else:
            N,Memoire,Lifetime,Popularity,Fitness=operation_micro(N,Memoire,Lifetime,Popularity,Fitness,Nu,alpha,mu,user,time)

        #On calcul le lifetime des memes éteints
        while 0 in N:
            Lifetime[N.index(0)] += time
            N[N.index(0)] = -2

        #On incremente time
        time+=1


    return N,Memoire,Lifetime,Popularity,Fitness,time



def simulation_micro(Nu,alpha,mu,timax):

    N,Memoire,Lifetime,Popularity,Fitness,init_time=init_micro(Nu,alpha,mu)


    time=init_time
    while time <= timax:
        user = np.random.randint(0,Nu)

        N,Memoire,Lifetime,Popularity,Fitness=operation_micro(N,Memoire,Lifetime,Popularity,Fitness,Nu,alpha,mu,user,time)


        #On calcul le lifetime des memes éteints
        while 0 in N:
            Lifetime[N.index(0)] += time
            N[N.index(0)] = -2

        #On incremente time
        time+=1

    return N,Memoire,Lifetime,Popularity,Fitness


# #Simulation

def sim(mu,temps):

    N_,Memoire_,Lifetime_,Popularity_,Fitness_ = simulation_micro(10**3,10,mu,temps)

    K = [meme for meme in range(len(N_)) if Lifetime_[meme]>0]

    Lifetime_k = [Lifetime_[i] for i in K]
    Popularity_k = [Popularity_[i] for i in K]
    Fitness_k = [int(Fitness_[i]*50)/50 for i in K]

    return Lifetime_k,Popularity_k,Fitness_k



##Popularity - Fitness

for mu_ in [0.1,0.2,0.4,0.6,0.8,1]:
    print(mu_)
    Lifetime_k,Popularity_k,Fitness_k = sim(mu=mu_,temps=2*10**5)
    fit_vu = []
    Popularity_mean = []
    for f in Fitness_k:
        if not f in fit_vu:
            fit_vu.append(f)
            A = [Popularity_k[i] for i in range(len(Popularity_k)) if Fitness_k[i]==f]
            Popularity_mean.append(np.mean(A))


    plt.plot(fit_vu,Popularity_mean,'.',label="mu="+str(mu_))

plt.xlabel("Fitness")
plt.ylabel("Popularity")
plt.legend()
plt.show()

##Lifetime - Fitness

for mu_ in [0.1,0.4,0.7,0.9]:
    print(mu_)
    Lifetime_k,Popularity_k,Fitness_k = sim(mu=mu_,temps=2*10**5)
    fit_vu = []
    Lifetime_mean = []
    for f in Fitness_k:
        if not f in fit_vu:
            fit_vu.append(f)
            A = [Lifetime_k[i] for i in range(len(Lifetime_k)) if Fitness_k[i]==f]
            Lifetime_mean.append(np.mean(A))


    plt.plot(fit_vu,Lifetime_mean,'.',label="mu="+str(mu_))
plt.xlabel("Fitness")
plt.ylabel("Lifetime")

plt.legend()
plt.show()



## P(popularity) - Popularity


for mu_ in [0.1,0.2,0.4,0.6,0.8]:
    print(mu_)
    Lifetime_k,Popularity_k,Fitness_k = sim(mu=mu_,temps=2*10**5)

    p_vu = []
    P_Popularity_vu = []
    for p in Popularity_k:
        if not p in p_vu:
            p_vu.append(p)
            P_Popularity_vu.append(Popularity_k.count(p)/len(Popularity_k))

    plt.plot(p_vu,P_Popularity_vu,'.',label="mu="+str(mu_))

plt.plot(Popularity_k+[100],[(Popularity_k[i])**(-2) for i in range(len(Popularity_k))]+[1/10000],label='x^(-2)')


plt.ylabel("P(Popularity)")
plt.xlabel("Popularity")
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.show()




## PDF (probabilities density function) - Lifetime
plt.close('all')

fig1 = plt.figure()

for mu_ in [0.1,0.2,0.4,0.6,0.8]:
    print(mu_)
    Lifetime_k,Popularity_k,Fitness_k = sim(mu=mu_,temps=2*10**5)

    #On discrétise les Lifetimes en echelle log
    Lifetime_discret = []
    PDF_discret=[]
    for l in Lifetime_k :
        ld = -1
        k=0
        while ld < 0:
            if k < 3:
                if l <= 10**(k+1):
                    ld = (l//(10**k))*(10**k)
                k+=1
            else :
                ld = (l//(10**k))*(10**k)

        if ld in Lifetime_discret :
            PDF_discret[Lifetime_discret.index(ld)]+=1
        else:
            Lifetime_discret.append(ld)
            PDF_discret.append(1)

    plt.plot(Lifetime_discret,np.array(PDF_discret)/len(Lifetime_k),'.',label="mu="+str(mu_))

plt.plot(Lifetime_discret,[(5*10**11)*Lifetime_discret[i]**(-3) for i in range(len(Lifetime_discret))],label='x^(-3)')

plt.ylabel("PDF")
plt.xlabel("Lifetime")
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.show()


















