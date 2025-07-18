 # Dinámica vı́trea en el aprendizaje de modelos de aprendizaje automático

Este repositorio contiene los archivos, datos y notebooks utilizados para los experimentos relacionados con el análisis de modelos de aprendizaje en MNIST y el modelo de Curie-Weiss.

## ESTRUCTURA
---
├── dataset/
│ └── *.gz # Archivos comprimidos de imágenes MNIST
│
├── files/
│ └── *.h5py # Resultados de entrenamientos, nombrados según la configuración usada
│
├── outputs_latex/
│ └── *.png / *.pdf # Imágenes generadas para la memoria en LaTeX
│
├── rbm.py # Definición del modelo RBM estándar
├── rbmg.py # RBM con modificaciones específicas (Gaussianas, etc.)
├── rbmg2.py # Variación alternativa de RBMG para experimentación
│
├── 1_Enero.ipynb
├── 2_Febrero.ipynb
├── 3_Marzo.ipynb
├── 4_Abril.ipynb
├── 5_Mayo.ipynb
├── 5_Mayo_2.ipynb
├── 6_Junio.ipynb
---

## NOTEBOOKS

### `1_Enero.ipynb`
- Entrenamientos con distintas configuraciones:
  - Número de nodos ocultos
  - Regularización y temperatura
  - Permutación de características
  - Análisis de susceptibilidad exponencial

### `2_Febrero.ipynb`
- Exploración con nodos ocultos gaussianos (`rbmg.py`)
- Comprobaciones preliminares
- Entrenamiento con un solo nodo oculto

### `3_Marzo.ipynb`
- Entrenamiento con RBMG óptimo
- Simulaciones Monte Carlo del modelo de Curie-Weiss
- Análisis de divergencia en el entrenamiento
- Dinámica del entrenamiento con un nodo oculto en MNIST
- Comparación con expresión analítica
- Generación de dataset de Curie-Weiss usando:
  - Metropolis
  - Heat Bath
- Red binaria (`rbm.py`, `rbmg.py`)

### `4_Abril.ipynb`
- Punto fijo para determinar magnetización
- Sampling del modelo Curie-Weiss
- Entrenamiento Curie-Weiss (`rbmg2.py`)
- Análisis a distintas temperaturas
- Enfriamiento temporal (tiempo a reversa)
- Entrenamiento sin nodo oculto
- Comparación de dinámicas

### `5_Mayo.ipynb` y `5_Mayo_2.ipynb`
- Sampling a lo largo de la matriz \( W \)
- Análisis de:
  - Magnetización
  - Susceptibilidad
  - Overlap
- Comparación con dinámica analítica
- Estudio de puntos fijos

### `6_Junio.ipynb`
- Análisis detallado de susceptibilidades
- Comparación numérica vs analítica

---



## CONVENCIÓN NOMBRE ARCHIVO PARÁMETROS ## 
 
	/files/[metodo]_n[N_hidden]_b[Batch_size]_l[learning_rate*]_k[gibbs_steps]_e[epochs**]_T[*, ***].h5

	* Al ser números menores que uno, solo se escribe la parte decimal (ej: 0.01 - l01)
	** Al ser mayor que 1000 se escribe con k (ej: 2000 - 2k)
	*** Temperatura opcional
	
	Ejemplo: RDMr_n50_b50_l01_k10_ek_T01.h5
	
	(La razon por lo que hago sí es para definirme la clase con los pámetros ya hechos y poder estudiarlos gráficamente)

## CARGAR RBM DE ARCHIVO ##

	Se necesitará definr el dataset y la escala de tiempos: 
	
	f = h5py.File(filename,'r')
	Ns = 10000
	device = 'cpu'
	mnist_trainset = datasets.MNIST('dataset/', train=True, download=True)
	D = mnist_trainset.data[:Ns,:,:].reshape(Ns,28*28).float().to(device) / 255.0
	D = (D > 0.3) * 1.0
	D = D.t()
	time = [0,1,2]
	for n in range (1,60):
		for m in range (1,60):
			t = 2**n + 2**m
			time.append(t)
	time = np.array(list(set(time)))
	time = np.sort(time)

## SAMPLING SIN ENTRENAMIENTO ## ======================================================
	Si se define la clase a partir de un archivo h5py, los features internos de la máquina se han inicializado a cero, así que habrá que cambiarlos por sus valores en el ultimo instante guardado en el archivo - función resetfeaures
	
## Código base 
Ns = 10000
device = 'cpu'
mnist_trainset = datasets.MNIST('dataset/', train=True, download=True)
D = mnist_trainset.data[:Ns,:,:].reshape(Ns,28*28).float().to(device) / 255.0
D = (D > 0.3) * 1.0
D = D.t()

time = [0,1,2]
for n in range (1,30):
    for m in range (1,30):
        t = 2**n + 2**m
        time.append(t)

time = np.array(list(set(time)))
time = np.sort(time)

filename = 'files/RDM_n50_b50_l01_k10_e2k.h5'
f = h5py.File(filename,'r') #leer


n_vis = D.shape[0]
n_hid = 50
dtype = torch.float
batch_size = 50
rdm = True
regu = True
lr = 0.01
gibbs_steps = 10
epoch_max = 2000
T = 0.01
RDMr = rbm.RBM(n_vis,n_hid,dtype,batch_size,time,rdm,regu,T,filename,lr,gibbs_steps,epoch_max)
