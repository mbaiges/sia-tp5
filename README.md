# Trabajo Práctico de Autoencoders y VAE para la materia Sistemas de Inteligencia Artificial

## Instalación

Para correr el programa debe ser necesario instalar python 3

[Descargar Python 3](https://www.python.org/downloads/)

Una vez instalado python, se necesitan la librería pyyaml.
Para eso, se debe tener instalado pip para python
La guía de instalación se encuentra en el siguiente link:

[Instalar Pip](https://tecnonucleous.com/2018/01/28/como-instalar-pip-para-python-en-windows-mac-y-linux/)

Una vez instalado pip, se debe correr, dentro de la carpeta del repositorio, el comando:

```python
pip install -r requirements.txt
```

## Guía de uso

### Configuración

Antes de ejectuar el programa, se debe modificar el archivo `config.yaml`.
En este archivo se deben configurar la ubicacion y el nombre del set de datos para los distintos ejercicios.

A continuación, se muestra un ejemplo de la configuración:

```yaml

# data folder
data_folder: data

# saves
saves_folder: saves

# data files

# fonts
fonts_h: font.h
fonts_training: fonts_training.json

# plotting
plotting: True

# multilayer config
momentum: True
momentum_mult: 0.7

# multilayer config
adaptative_lr: False

# progress bar
progress_bar: True

# vae config
images_folder: developer-faces
images_shape:
  width: 16
  height: 16


```

### Ejecución

Finalmente, para correr los distintos puntos del trabajo se debe ejecutar uno de los siguientes comandos para ejecutar el ejercicio deseado:

```python

python .\exercise.py

python .\ej1a.py

python .\ej1a_usual_matches.py:

python .\ej1b.py

python .\ej2.py

python .\ej2_rt.py

```

* `exercise.py`:
    Genera nuevas componentes como combinación lineal de las originales, y muestra a los paises en un mapa ubicados en función de la 1ra y la 2da componente nueva

* `ej1a.py`:
    Crea una red de Kohonen y realiza el ordenamiento de los items del conjunto de entrada en una matriz de neuronas. Muestra los graficos de:         
    * La cantidad de items que cayeron en cada neurona durante todo el proceso
    * Una matriz U del promedio de la distancia entre el vector de pesos del nodo y sus nodos vecinos
    * Una matriz con la cantidad de items que cayeron en cada neurona en la ultima epoca


* `ej1a_usual_matches.py`:
    Genera una red de Kohonen y realiza el ordenamiento de los items del conjunto de entrada multiples veces. En cada iteración observamos que items quedaron agrupados en la misma neurona y lo almacenamos. Al final de todas las iteraciones, mostramos un grafico que muestra cuantas veces compartieron la posicion entre pares de items de mi conjunto de entrada

* `ej1b.py`:
    Realiza la busqueda de la 1ra componente utilizando un perceptrón lineal simple con la regla de Oja

* `ej2.py`:
    Utilizando una red de Hopfield, almacena 4 patrones de letras ortogonales entre sí y para cada una de ellas, realiza una prueba agregando ruido al patrón original desde un 0% hasta un 100% con un paso del 5% durante 1000 iteraciones para cada una. Finalmente muestra un grafico para cada letra donde se ve la cantidad de aciertos, la cantidad de estados espureos y la cantidad de aciertos erróneos.

* `ej2_rt.py`:
    Utiliza una red de Hopfield, y muesta, a tiempo real, el desarrollo del algoritmo para un caracter con ruido
