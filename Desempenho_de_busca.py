#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import math
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap


# In[2]:


def BS(array_length, key): #Binary Search
    #retorna o numero de voltas que o algoritimo faz para encontrar o item via BS
    
    first = 0
    last = array_length
    mid = round(array_length/2)
    value = 0
    
    while(first<=last):
        value = value + 1
        
        if(mid == key):
            return value
        if(mid > key):
            last = mid-1
        else:
            first = mid+1
            
        #mid = round((last + first)/2)
        mid =  round(first + ((last-first) / 2))
        
    return 0
    


# In[3]:


def BS_max(array_length): #Binary Search
    #Retorna o maior numero de voltas que a BS pode fazer em uma lista de comprimento X
    return (math.ceil(math.log(array_length,2))) # log de 2 elevado a X


# In[4]:


def BS_performance(array_length): #Binary Search
    #Mede o desempenho da BS através do número de voltas que ela faz para encontrar cada item na lista
    
    performance = np.zeros(array_length)
    for i in range(0, array_length):
        if(performance[i] == 0):
            performance[i] =  BS(array_length, i)
        
    return performance


# In[5]:


def LS_performance(a_array_length): #Linear Search
    #Mede o desempenho da busca linear através do número de voltas que ela faz para encontrar cada item na lista
    return np.arange(1,a_array_length+1,1)
    


# In[19]:


def g_BS_performance(base, amount, n_array, label): #Binary Search
    #Exibe um gráfico com o desempenho da BS em listas de diferentes tamanhos 
    #A escala de cores tem como base o maior e menor valor dentre os gráficos
    
    title = 'binary_search_performance_X' + str(n_array) + '_' + str(base) + '_' + str(base+amount*(n_array-1)) + str(label) + '.png'
    
    #cria um novo mapa de cores
    colors1 = cm.get_cmap('Blues', 128)
    colors = np.vstack((colors1(np.linspace(1, 0, 128))))
    cmp = ListedColormap(colors)

    #gerencia a distribuição de cores pelo grafico
    percentage = (100/n_array)/100
    percent_x = 1 - percentage
    
    max_size = base+(amount*(n_array-1))
    scale = 5
    
    if(n_array>1): #cria varios gráficos
        
        if(max_size/10 > 1):
            fig, axes = plt.subplots(1,n_array, constrained_layout=False, figsize=(n_array,max_size/scale))
        else:
            fig, axes = plt.subplots(1,n_array, constrained_layout=False, figsize=(n_array,3))
            
        for current_ax in axes:
            
            newcmp = ListedColormap(cmp(np.linspace(1, percent_x, 256))) #adiciona as cores
            percent_x = percent_x - percentage
            array = BS_performance(base)
            current_ax.pcolormesh(np.row_stack(array),cmap=newcmp)
            
            if(label):
                #legenda
                current_ax.set_yticklabels(array)
                current_ax.set_yticks(np.arange(1, base+1, 1.0))
                current_ax.set_xticks(np.arange(0, 0, 1.0))
                
            else:
                #set_axis_off()
                current_ax.set_xticks(np.arange(0, 0, 1.0))
                current_ax.set_yticks(np.arange(0, 0, 1.0))
            
            current_ax.set_title(base)
            base = base + amount
    
    else: #cria apenas um gráfico
        
        if(max_size/10 > 1):
            fig, axes = plt.subplots(figsize=(n_array,max_size/scale))
        else:
            fig, axes = plt.subplots(figsize=(n_array,3))
            
        newcmp = ListedColormap(cmp(np.linspace(1, percent_x, 256)))  
        array = BS_performance(base)
        axes.pcolormesh(np.row_stack((array)),cmap=newcmp)
        
        if(label):
            #legenda
                axes.set_yticklabels(array)
                axes.set_yticks(np.arange(1, base+1, 1.0))
                axes.set_xticks(np.arange(0, 0, 1.0))
        else:
            #set_axis_off()
                axes.set_xticks(np.arange(0, 0, 1.0))
                axes.set_yticks(np.arange(0, 0, 1.0))
        
        
        

        #colorbar
        '''
        im=axes.pcolormesh(np.row_stack((BS_performance(base))),cmap=newcmp)
        divider = make_axes_locatable(axes)
        cax = divider.append_axes("right", size="10%", pad=0.05)
        plt.colorbar(im, cax=cax)'''
        
        axes.set_title(base)
    
    plt.tight_layout()
    plt.savefig((title), format='png')
    #plt.show() 


# In[20]:


def g_LS_performance(base, amount, n_array, label):  #Linear Search
    #Exibe um gráfico com o desempenho da LS em listas de diferentes tamanhos 
    #A escala de cores tem como base o maior e menor valor dentre os gráficos
    
    title = 'linear_search_performance_X' + str(n_array) + '_' + str(base) + '_' + str(base+amount*(n_array-1)) + str(label) + '.png'
    
    #cria um novo mapa de cores
    colors1 = cm.get_cmap('ocean', 128)
    colors2 = cm.get_cmap('RdYlGn', 128)

    colors = np.vstack((colors1(np.linspace(1, 0.08, 128)),colors2(np.linspace(1, 0, 128))))
    cmp = ListedColormap(colors)
    
    #gerencia a distribuição de cores pelo grafico
    percentage = (100/n_array)/100
    percent_x = 1 - percentage
    
    max_size = base+(amount*(n_array-1))
    scale = 5
    
    if(n_array>1): #cria varios gráficos
        
        if(max_size/10 > 1):
            fig, axes = plt.subplots(1,n_array, constrained_layout=False, figsize=(n_array,max_size/scale))
        else:
            fig, axes = plt.subplots(1,n_array, constrained_layout=False, figsize=(n_array,3))

        for current_ax in axes:
            
            newcmp = ListedColormap(cmp(np.linspace(1-percent_x, 0, 256))) #adiciona as cores
            percent_x = percent_x - percentage
            array = LS_performance(base)
            current_ax.pcolormesh(np.row_stack((array)),cmap=newcmp)
               
            
            if(label):
                #legenda
                current_ax.set_yticklabels(array[::-1])
                current_ax.set_yticks(np.arange(min(array), max(array)+1, 1.0))
                current_ax.set_xticks(np.arange(0, 0, 1.0))
            else:
                #set_axis_off()
                current_ax.set_xticks(np.arange(0, 0, 1.0))
                current_ax.set_yticks(np.arange(0, 0, 1.0))
            
            current_ax.set_title(base)
            base = base + amount
            
    else:  #cria apenas um gráfico
        
        if(max_size/10 > 1):
            fig, axes = plt.subplots(figsize=(n_array,max_size/scale))
        else:
            fig, axes = plt.subplots(figsize=(n_array,3))
            
        newcmp = ListedColormap(cmp(np.linspace(1,0, 256))) #adiciona as cores   
        array = LS_performance(base)
        axes.pcolormesh(np.row_stack(array),cmap=newcmp)
        
        if(label):
            #legenda
            axes.set_yticklabels(array[::-1])
            axes.set_yticks(np.arange(min(array), max(array)+1, 1.0))
            axes.set_xticks(np.arange(0, 0, 1.0))
        else:
            #set_axis_off()
            axes.set_xticks(np.arange(0, 0, 1.0))
            axes.set_yticks(np.arange(0, 0, 1.0))
        
        

        axes.set_title(base)

    plt.tight_layout()   
    plt.savefig((title), format='png')
    #plt.show()


# In[21]:


def performance_comparison(a_array_length, a_search_type, label): #Performance Comparison
    #Faz uma comparação entre os diferentes tipos de busca
    #A escala de cores tem como base o tamanho do array
    
    if(len(a_array_length) != len(a_search_type)):
        return -1
    
    
    #cria um novo mapa de cores
    colors1 = cm.get_cmap('ocean', 128)
    colors2 = cm.get_cmap('RdYlGn', 128)

    colors = np.vstack((colors2(np.linspace(0, 1, 128)),colors1(np.linspace(0.1, 1, 128))))
    cmp = ListedColormap(colors)
    
    max_size = max(a_array_length)
    scale = 5
    
    if(max_size/10 > 1):
        fig, axes = plt.subplots(1,len(a_search_type), constrained_layout=False, figsize=(len(a_array_length),max_size/scale))
    else:
        fig, axes = plt.subplots(1,len(a_search_type), constrained_layout=False, figsize=(len(a_array_length),3))

    
    i=0 #controla o 'a_array_length' e o 'a_search_type'
    for current_ax in axes:
        
        if(a_search_type[i]=='BS'): #Binary Search
            array = BS_performance(a_array_length[i])         
            percent_x = (100*BS_max(a_array_length[i])/a_array_length[i])*0.01 #ajusta a escala de cores
            newcmp = ListedColormap(cmp(np.linspace(1, 1-percent_x, 256)))
            
            if(label):
                #legenda
                current_ax.set_yticklabels(array)
                current_ax.set_yticks(np.arange(1, len(array)+1, 1.0))
                current_ax.set_xticks(np.arange(0, 0, 1.0))
                
            else:
                #set_axis_off()
                current_ax.set_xticks(np.arange(0, 0, 1.0))
                current_ax.set_yticks(np.arange(0, 0, 1.0))
            
        elif(a_search_type[i]=='LS'): #Linear Search
            array = LS_performance(a_array_length[i])
            percent_x = (100/a_array_length[i])/100 #ajusta a escala de cores
            newcmp = ListedColormap(cmp(np.linspace(percent_x, 1, 256)))
            
            if(label):
                #legenda
                current_ax.set_yticklabels(array[::-1])
                current_ax.set_yticks(np.arange(min(array), max(array)+1, 1.0))
                current_ax.set_xticks(np.arange(0, 0, 1.0))
            else:
                #set_axis_off()
                current_ax.set_xticks(np.arange(0, 0, 1.0))
                current_ax.set_yticks(np.arange(0, 0, 1.0))
       
        current_ax.pcolormesh(np.row_stack(array),cmap=newcmp)
        current_ax.set_title(a_search_type[i] + " " + str(a_array_length[i]))    
        i = i+1
    
    title = 'performance_comparison_'
    for i in range(0, len(a_array_length)):
        title += a_search_type[i]
        title += str(a_array_length[i])
        title += '_'
    
    plt.tight_layout()
    
    
    title += str(label)
    title += '.png'
    plt.savefig((title), format='png')
    #plt.show()   


# In[9]:


def g_sum_performance(base, amount, g_length ,search_type): #SUM
    #Exibe um gráfico com o valor total de voltas que a busca pode fazer em listas de diferentes tamanhos   
    
    title = 'sum' + '_' + search_type + '_' + str(base) + '_' + str(base+amount*(g_length-1)) + '.png'
    
    
    g_X = pd.Series(g_length) # Tamanho da lista
    g_Y = pd.Series(g_length) # Total de espera
    
    fig, axes = plt.subplots()
    axes.set(xlabel='Array Length', ylabel='Performance (Sum)')
    axes.grid()
    
    for i in range(0,g_length):
        
        g_X[i] = base
        
        if(search_type == 'BS'): #Binary Search
            g_Y[i] = sum(BS_performance(base))
            plt.title('Binary Search')
        elif(search_type == 'LS'): #Linear Search
            g_Y[i] = sum(LS_performance(base))
            plt.title('Linear Search')
        else:
            return -1
        
        base = base + amount
   
    axes.plot(g_X, g_Y)
    
    plt.savefig((title), format='png')
    #plt.show()


# In[10]:


def mean_performance(array_length, search_type): #Average
    #Faz a média do desempenho da busca
    
    if(search_type == 'BS'): #Binary Search
        return (sum(BS_performance(array_length))/array_length)
        
    elif(search_type == 'LS'): #Linear Search
        return (sum(LS_performance(array_length))/array_length)
        
    else:
        return -1


# In[11]:


def g_mean_performance(base, amount, g_length, search_type): #Average
    #Exibe um gráfico com o desempenho médio da em listas de diferentes tamanhos
    
    title = 'average' + '_' + search_type + '_' + str(base) + '_' + str(base+amount*(g_length-1)) + '.png'
    
    g_X = pd.Series(g_length) # Tamanho da lista
    g_Y = pd.Series(g_length) # Total de espera
       
    fig, axes = plt.subplots()
    axes.set(xlabel='Array Length', ylabel='Performance (Average)')
    axes.grid()
    
    for i in range(0,g_length):
        g_X[i] = base
        
        if(search_type == 'BS'): #Binary Search
            g_Y[i] = mean_performance(base, search_type)
            plt.title('Binary Search')
        elif(search_type == 'LS'): #Linear Search
            g_Y[i] = mean_performance(base, search_type)
            plt.title('Linear Search')
        else:
            return -1
        
        base = base + amount

    axes.plot(g_X, g_Y)
    
    plt.savefig((title), format='png')
    #plt.show()


# In[22]:


g_BS_performance(10, 1, 1, True)
g_BS_performance(20, 1, 1, True)
g_BS_performance(10, 10, 5, True)
g_BS_performance(10, 10, 5, False)

g_LS_performance(10, 1, 1, True)
g_LS_performance(20, 1, 1, True)
g_LS_performance(10, 10, 5, True)
g_LS_performance(10, 10, 5, False)

performance_comparison([40,40,20,20,10,10], ['BS','LS','BS','LS','BS','LS'], True)
performance_comparison([40,40,20,20,10,10], ['BS','LS','BS','LS','BS','LS'], False)

g_sum_performance(10, 10, 50,'BS')
g_sum_performance(10, 10, 50,'LS')

g_mean_performance(10, 10, 100,'BS')
g_mean_performance(10, 10, 100,'LS')

