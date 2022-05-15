import numpy as np
import matplotlib.pyplot as plt


##Initialisation : degree = 1, Nu = 10**3, alpha = 10 f=1


def operation_micro(N,Memoire,Lifetime,Popularity,Nu,alpha,mu,user,time):

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


    #Transmission d'1 meme
    else:
        receiver = user-1

        meme = np.random.choice(Memoire[user])
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
    return N,Memoire,Lifetime,Popularity



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

    time = 0

    while sum([len(Memoire[i]) for i in range(len(Memoire))]) < alpha*Nu :


        user = np.random.randint(0,Nu)

        #Si user a la mémoire vide : il crée un meme
        if Memoire[user]==[]:
            Memoire[user].append(len(N))
            N.append(alpha)
            Lifetime.append(-time)
            Popularity.append(1)


        #Sinon on  procéde noramlement
        else:
            N,Memoire,Lifetime,Popularity=operation_micro(N,Memoire,Lifetime,Popularity,Nu,alpha,mu,user,time)

        #On calcul le lifetime des memes éteints
        while 0 in N:
            Lifetime[N.index(0)] += time
            N[N.index(0)] = -2

        #On incremente time
        time+=1


    return N,Memoire,Lifetime,Popularity,time



def simulation_micro2(Nu,alpha,mu,timax):

    N,Memoire,Lifetime,Popularity,init_time=init_micro(Nu,alpha,mu)


    time=init_time
    while time <= timax:
        user = np.random.randint(0,Nu)

        N,Memoire,Lifetime,Popularity=operation_micro(N,Memoire,Lifetime,Popularity,Nu,alpha,mu,user,time)


        #On calcul le lifetime des memes éteints
        while 0 in N:
            Lifetime[N.index(0)] += time
            N[N.index(0)] = -2

        #On incremente time
        time+=1

    return N,Memoire,Lifetime,Popularity


# #Simulation

def sim2(Nu,mu,temps):

    N_,Memoire_,Lifetime_,Popularity_ = simulation_micro2(Nu,10,mu,temps)

    K = [meme for meme in range(len(N_)) if Lifetime_[meme]>0]

    Lifetime_k = [Lifetime_[i] for i in K]#On discrétise les Lifetimes
    Popularity_k = [Popularity_[i] for i in K]

    return K,Lifetime_k,Popularity_k


##Vérification F(l) ~ l**(-2)


Nu_=10**3
timax_ = 10**5

for mu_ in [0.01,0.05,0.1,0.2,0.4,0.6]:
    print(mu_)
    K,Lifetime_k,Popularity_k=sim2(Nu_,mu_,timax_)
    print('K ~',len(K))
    # # Density Lifetime (purely diffuse dynamics) m~0,f=1

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

    plt.plot(Lifetime_discret,np.array(PDF_discret)/len(K),'.',label="mu="+str(mu_))






def F(l,alpha=10):
    return alpha*l**(-2)

plt.plot(Lifetime_discret,[F(l) for l in Lifetime_discret],label="alpha*l^(-2)")
plt.plot(Lifetime_discret,[l**(-2) for l in Lifetime_discret],label="l^(-2)")


plt.ylabel("PDF")
plt.xlabel("Lifetime")
plt.xscale('log')
plt.yscale('log')

plt.legend()
plt.show()



## Density Lifetime (pure drift) m~0,f=1, Nu -> +00 (non terminé)
fig3 = plt.figure()
mu_ = 0.001
timax_ = 10**5
Nu_ = 10**4


K,Lifetime_k,Popularity_k=sim2(Nu_,mu_,timax_)
print('ok')

print('K ~',len(K))
Lifetime_discret = []
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

    Lifetime_discret.append(ld)



Lifetime_density = []
l_vu = []
for l in Lifetime_discret:
    print(len(l_vu))
    if not l in l_vu:
        l_vu.append(l)
        Lifetime_density.append(Lifetime_discret.count(l)/len(K))


def F1(l,alpha=10):
    return alpha*l**(-2)

def F2(l,mu,f=1,alpha=10):
    td = ((1-(1-mu)*f)*(alpha+1))**(-1)
    return alpha*l**(-2)*np.exp(-l/td)


plt.plot(l_vu,[Lifetime_density[i]/F2(l_vu[i],mu_) for i in range(len(l_vu))],'*',label="alpha*l-2*exp")


plt.xscale('log')
plt.yscale('log')
plt.xlabel=('Lifetime l')
plt.ylabel('Density F(l)/label')
plt.legend()
plt.show()


"""Nous n'avons pas eu le temps de terminer ce code, il y a un problème sur la petitesse des valeurs rendues par F2 que pyzo assimile à des 0."""

























