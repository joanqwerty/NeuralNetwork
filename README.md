# Red Neuronal para compuerta XOR

Interfaz CLI para entrenar una red neuronal que resuelva la compuerta lógica XOR.

## Interfaz

Al iniciar te aparece un menú con las opciones que puedes ejecutar:

1. Entrenar el modelo
2. Hacer una predicción
3. Eliminar Modelo

### Entrenar el modelo

Manejo de parametros para entrenar la red, y muestra las epocas de entrenamiento.

### Hacer predicción

Ver las predicciones haciendo uso de la red.

### Eliminar modelo

Al entrenar la red se guarda el modelo, con está opcion se pude eliminar para empezar de nuevo el proceso.

## compilación y ejecución

Instalar las dependencias necesarias.

```bash
    sudo apt update
    sudo apt install libarmadillo-dev cmake
```

Crear carpeta y compilar.

```bash
    mkdir build
    cd build
    cmake ..
    make
    ./NN
```
