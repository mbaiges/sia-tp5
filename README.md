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

En el archivo de configuración se debe especificar:
 * `data_folder` : la carpeta donde estan los datos guardados que seran utilizados para entrenar a la red
 * `saves_folder` : la carpeta donde volcar los pesos de la red entrenada
 * `fonts_h` : nombre del archivo .h de fuentes en binario provisto por la catedra
 * `fonts_training` : un archivo con los fonts_h pasados de hexadecimal  a entero en formato JSON
 * `plotting`: un valor booleano para indicar si se quiere graficar
 * `momentum y momentum_mult` : un valor booleano para indicar si se quiere utilizar momentum. Y el porcentaje que se quiere utilizar
 * `adaptative_lr` :  un valor booleano para indicar si se quiere utilizar un learning rate adaptativo
 * `images_folder y images_shape`: La carpeta donde se deben encontrar las imagenes para el VAE y el tamaño al que se las desea comprimir
 

### Ejecución

Finalmente, para correr los distintos puntos del trabajo se debe ejecutar uno de los siguientes comandos para ejecutar el ejercicio deseado:

```python

python .\ej1a_train.py

python .\ej1a1_use.py

python .\ej1a3_use.py

python .\ej1a4_use.py

python .\ej1b_train.py

python .\ej1b_use.py

python .\ej2_images.py

python .\ej1a_greedy_layer_train.py

python .\ej1a_greedy_layer_use.py

python .\ej1a_greedy_layer.py

```

*Aclaración: Las secuencias de comandos no ejecutarán el entrenado por el código anterior `_train`, para esto hay que modificar en los `_use` el nombre del archivo a utilizar por el nombre del generado.*

* `ej1a_train.py, ej1b_train.py`:
    Corresponden a las partes "a" y "b" del punto 1 del trabajo y se deben correr para entrenar a la red de la forma deseada. El archivo con la información del entrenamiento se guarda en la carpeta especificada en `saves` del archivo de configuracion yaml. La red que se desee crear debe ser modificada en la funcion main de cualquiera de estos 2 archivos

* `ej1a_greedy_layer_train.py`:
    Similar a ej1a_train pero utiliza una red que entrena por niveles de afuera hacia adentro en redes neuronales separadas. Permite almacenar distintas capas en archivos separados, y entrenar las siguientes capas eligiendo que capas pre-entrenadas queremos.

* `ej1a1_use.py, ej1a3_use.py, ej1a4_use.py, ej1b_use.py, ej2_images.py`:
    Utiliza las redes correspondientes creadas con los archivos `_train` y realiza la tarea correspondiente a cada ejercicio.

* `ej1a_greedy_layer_use.py`:
    Permite utilizar redes correspondientes a distintas capas provenientes de distintos archivos y evaluar el error del autoencoder generado utilizando todas estas capas.

* `ej1a_greedy_layer.py`:
    Permite definir una estructura de capas (todas propagando el error con la función de tangente hiperbólica), entrenar cada par de capas individualmente, y luego generar el perceptrón multicapa y almacenarlo para luego evaluarlo utilizando por ejemplo el archivo `ej1a1_use.py`.