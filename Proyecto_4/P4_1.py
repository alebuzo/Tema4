

# ---
# 
# ## Universidad de Costa Rica
# ### Escuela de Ingeniería Eléctrica
# #### IE0405 - Modelos Probabilísticos de Señales y Sistemas
# 
# ---
# 
# * Estudiantes: **Limber Nayib Benavides Araya, Alexa Mariana Carmona Buzo, Carlos Daniel Gómez Vargas y Bryan Josué Ruiz García**
# * Carné (respectivamente): **B12345, B12345, B93319 y B66378**
# * Grupo: **1/2**
# 
# ---
# # `P4` - *Modulación digital IQ*
# 
# > La modulación digital es una de las aplicaciones del análisis de procesos estocásticos, y es parte de los sistemas digitales de comunicación. Este proyecto presenta una introdución a tópicos fundamentales de la ingeniería de comunicaciones para simular un sistema de transmisión de imágenes de baja resolución.
# 
# ---

# ## 3. - Simulando un sistema de comunicaciones 
# 

# A continuación se especificarán las funciones desarrolladas en la simulación del sistema (considerar que en telecomunicaciones se utiliza **Tx** para referirse a transmisión y **Rx** a recepción):


from PIL import Image
import numpy as np

def fuente_info(imagen):
    '''Una función que simula una fuente de
    información al importar una imagen y 
    retornar un vector de NumPy con las 
    dimensiones de la imagen, incluidos los
    canales RGB: alto x largo x 3 canales

    :param imagen: Una imagen en formato JPG
    :return: un vector de pixeles
    '''
    img = Image.open(imagen)
    
    return np.array(img)


#----------------------------------------------------------------------------------

import numpy as np

def rgb_a_bit(array_imagen):
    '''Convierte los pixeles de base 
    decimal (de 0 a 255) a binaria 
    (de 00000000 a 11111111).

    :param imagen: array de una imagen 
    :return: Un vector de (1 x k) bits 'int'
    '''
    # Obtener las dimensiones de la imagen
    x, y, z = array_imagen.shape
    
    # Número total de elementos (pixeles x canales)
    n_elementos = x * y * z

    # Convertir la imagen a un vector unidimensional de n_elementos
    pixeles = np.reshape(array_imagen, n_elementos)

    # Convertir los canales a base 2
    bits = [format(pixel, '08b') for pixel in pixeles]
    bits_Rx = np.array(list(''.join(bits)))
    
    return bits_Rx.astype(int)


# #### 3.1.3. - Esquema de modulación BPSK



import numpy as np

def modulador(bits, fc, mpp):
    '''Un método que simula el esquema de 
    modulación digital BPSK.

    :param bits: Vector unidimensional de bits
    :param fc: Frecuencia de la portadora en Hz
    :param mpp: Cantidad de muestras por periodo de onda portadora
    :return: Un vector con la señal modulada
    :return: Un valor con la potencia promedio [W]
    :return: La onda portadora c(t)
    :return: La onda cuadrada moduladora (información)
    '''
    # 1. Parámetros de la 'señal' de información (bits)
    N = len(bits) # Cantidad de bits

    # 2. Construyendo un periodo de la señal portadora c(t)
    Tc = 1 / fc  # periodo [s]
    t_periodo = np.linspace(0, Tc, mpp)  # mpp: muestras por período
    portadora = np.sin(2*np.pi*fc*t_periodo)

    # 3. Inicializar la señal modulada s(t)
    t_simulacion = np.linspace(0, N*Tc, N*mpp) 
    senal_Tx = np.zeros(t_simulacion.shape)
    moduladora = np.zeros(t_simulacion.shape)  # (opcional) señal de bits
 
    # 4. Asignar las formas de onda según los bits (BPSK)
    for i, bit in enumerate(bits):
        if bit == 1:
            senal_Tx[i*mpp : (i+1)*mpp] = portadora
            moduladora[i*mpp : (i+1)*mpp] = 1
        else:
            senal_Tx[i*mpp : (i+1)*mpp] = portadora * -1
            moduladora[i*mpp : (i+1)*mpp] = 0
    
    # 5. Calcular la potencia promedio de la señal modulada
    P_senal_Tx = (1 / (N*Tc)) * np.trapz(pow(senal_Tx, 2), t_simulacion)
    
    return senal_Tx, P_senal_Tx, portadora, moduladora  


# #### 3.1.4. - Construcción de un canal con ruido AWGN

import numpy as np

def canal_ruidoso(senal_Tx, Pm, SNR):
    '''Un bloque que simula un medio de trans-
    misión no ideal (ruidoso) empleando ruido
    AWGN. Pide por parámetro un vector con la
    señal provieniente de un modulador y un
    valor en decibelios para la relación señal
    a ruido.

    :param senal_Tx: El vector del modulador
    :param Pm: Potencia de la señal modulada
    :param SNR: Relación señal-a-ruido en dB
    :return: La señal modulada al dejar el canal
    '''
    # Potencia del ruido generado por el canal
    Pn = Pm / pow(10, SNR/10)

    # Generando ruido auditivo blanco gaussiano (potencia = varianza)
    ruido = np.random.normal(0, np.sqrt(Pn), senal_Tx.shape)

    # Señal distorsionada por el canal ruidoso
    senal_Rx = senal_Tx + ruido

    return senal_Rx


# #### 3.1.5. - Esquema de demodulación



import numpy as np

def demodulador(senal_Rx, portadora, mpp):
    '''Un método que simula un bloque demodulador
    de señales, bajo un esquema BPSK. El criterio
    de demodulación se basa en decodificación por 
    detección de energía.

    :param senal_Rx: La señal recibida del canal
    :param portadora: La onda portadora c(t)
    :param mpp: Número de muestras por periodo
    :return: Los bits de la señal demodulada
    '''
    # Cantidad de muestras en senal_Rx
    M = len(senal_Rx)

    # Cantidad de bits (símbolos) en transmisión
    N = int(M / mpp)

    # Vector para bits obtenidos por la demodulación
    bits_Rx = np.zeros(N)

    # Vector para la señal demodulada
    senal_demodulada = np.zeros(senal_Rx.shape)

    # Pseudo-energía de un período de la portadora
    Es = np.sum(portadora * portadora)

    # Demodulación
    for i in range(N):
        # Producto interno de dos funciones
        producto = senal_Rx[i*mpp : (i+1)*mpp] * portadora
        Ep = np.sum(producto) 
        senal_demodulada[i*mpp : (i+1)*mpp] = producto

        # Criterio de decisión por detección de energía
        if Ep > 0:
            bits_Rx[i] = 1
        else:
            bits_Rx[i] = 0

    return bits_Rx.astype(int), senal_demodulada


# #### 3.1.6. - Reconstrucción de la imagen



import numpy as np

def bits_a_rgb(bits_Rx, dimensiones):
    '''Un blque que decodifica el los bits
    recuperados en el proceso de demodulación

    :param: Un vector de bits 1 x k 
    :param dimensiones: Tupla con dimensiones de la img.
    :return: Un array con los pixeles reconstruidos
    '''
    # Cantidad de bits
    N = len(bits_Rx)

    # Se reconstruyen los canales RGB
    bits = np.split(bits_Rx, N / 8)

    # Se decofican los canales:
    canales = [int(''.join(map(str, canal)), 2) for canal in bits]
    pixeles = np.reshape(canales, dimensiones)

    return pixeles.astype(np.uint8)


# ### 3.2. - Simulación del sistema de comunicaciones con modulación BPSK
# 
# **Nota**: esta simulación tarda un poco y quizá hay que hacerla dos veces.


import numpy as np
import matplotlib.pyplot as plt
import time

# Parámetros
fc = 5000  # frecuencia de la portadora
mpp = 20   # muestras por periodo de la portadora
SNR = -5   # relación señal-a-ruido del canal

# Iniciar medición del tiempo de simulación
inicio = time.time()

# 1. Importar y convertir la imagen a trasmitir
imagen_Tx = fuente_info('arenal.jpg')
dimensiones = imagen_Tx.shape

# 2. Codificar los pixeles de la imagen
bits_Tx = rgb_a_bit(imagen_Tx)

# 3. Modular la cadena de bits usando el esquema BPSK
senal_Tx, Pm, portadora, moduladora = modulador(bits_Tx, fc, mpp)

# 4. Se transmite la señal modulada, por un canal ruidoso
senal_Rx = canal_ruidoso(senal_Tx, Pm, SNR)

# 5. Se desmodula la señal recibida del canal
bits_Rx, senal_demodulada = demodulador(senal_Rx, portadora, mpp)

# 6. Se visualiza la imagen recibida 
imagen_Rx = bits_a_rgb(bits_Rx, dimensiones)
Fig = plt.figure(figsize=(10,6))

# Cálculo del tiempo de simulación
print('Duración de la simulación: ', time.time() - inicio)

# 7. Calcular número de errores
errores = sum(abs(bits_Tx - bits_Rx))
BER = errores/len(bits_Tx)
print('{} errores, para un BER de {:0.4f}.'.format(errores, BER))

# Mostrar imagen transmitida
ax = Fig.add_subplot(1, 2, 1)
imgplot = plt.imshow(imagen_Tx)
ax.set_title('Transmitido')

# Mostrar imagen recuperada
ax = Fig.add_subplot(1, 2, 2)
imgplot = plt.imshow(imagen_Rx)
ax.set_title('Recuperado')
Fig.tight_layout()

plt.imshow(imagen_Rx)

#------------------------------------------------------------------------


import matplotlib.pyplot as plt

# Visualizar el cambio entre las señales
fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, sharex=True, figsize=(14, 7))

# La onda cuadrada moduladora (bits de entrada)
ax1.plot(moduladora[0:600], color='r', lw=2) 
ax1.set_ylabel('$b(t)$')

# La señal modulada por BPSK
ax2.plot(senal_Tx[0:600], color='g', lw=2) 
ax2.set_ylabel('$s(t)$')

# La señal modulada al dejar el canal
ax3.plot(senal_Rx[0:600], color='b', lw=2) 
ax3.set_ylabel('$s(t) + n(t)$')

# La señal demodulada
ax4.plot(senal_demodulada[0:600], color='m', lw=2) 
ax4.set_ylabel('$b^{\prime}(t)$')
ax4.set_xlabel('$t$ / milisegundos')
fig.tight_layout()
plt.show()


# ### 3.3. - Modulación IQ


import numpy as np

def modulador8(bits,fc,mmp):
    '''Un método que simula el esquema de 
    modulación digital 8-PSK.

    :param bits: Vector unidimensional de bits
    :param fc: Frecuencia de la portadora en Hz
    :param mpp: Cantidad de muestras por periodo de onda portadora
    :return: Un vector con la señal modulada
    :return: Un valor con la potencia promedio [W]
    :return: La onda portadora c(t)
    :return: La onda cuadrada moduladora (información)
    '''

    # 1. Parámetros de la señal
    N = len(bits) # Cantidad de bits

    # 2. Construyendo un periodo de la señal portadora c(t)
    Tc = 1 / fc  # periodo [s]
    t_periodo = np.linspace(0, Tc, mpp)
    # Portadora 1 de s(t)
    portadora1 = np.cos(2*np.pi*fc*t_periodo)
    # Portadora 2 de s(t)
    portadora2 = np.sin(2*np.pi*fc*t_periodo) 

    # 3. Inicializar la señal modulada s(t)
    t_simulacion = np.linspace(0, N*Tc, N*mpp) 
    senal_Tx1 = np.zeros(t_simulacion.shape)
    senal_Tx2 = np.zeros(t_simulacion.shape)
    
    # 4. Asignar las formas de onda según los bits (8-PSK)
    for i in range(len(bits)):
        if (i%3 == 0):
            if(bits[i]==1 and bits[i+1]==1 and bits[i+2]==1):
                senal_Tx1[i*mpp : (i+1)*mpp] = portadora1*1
                senal_Tx2[i*mpp : (i+1)*mpp] = portadora2*0
            
            if(bits[i]==1 and bits[i+1]==1 and bits[i+2]==0):
                senal_Tx1[i*mpp : (i+1)*mpp] = portadora1*np.sqrt(2)/2
                senal_Tx2[i*mpp : (i+1)*mpp] = portadora2*np.sqrt(2)/2
            
            if(bits[i]==0 and bits[i+1]==1 and bits[i+2]==0):
                senal_Tx1[i*mpp : (i+1)*mpp] = portadora1*0
                senal_Tx2[i*mpp : (i+1)*mpp] = portadora2*1
            
            if(bits[i]==0 and bits[i+1]==1 and bits[i+2]==1):
                senal_Tx1[i*mpp : (i+1)*mpp] = portadora1*-np.sqrt(2)/2
                senal_Tx2[i*mpp : (i+1)*mpp] = portadora2*np.sqrt(2)/2
                
            if(bits[i]==0 and bits[i+1]==0 and bits[i+2]==1):
                senal_Tx1[i*mpp : (i+1)*mpp] = portadora1*-1
                senal_Tx2[i*mpp : (i+1)*mpp] = portadora2*0
                
            if(bits[i]==0 and bits[i+1]==0 and bits[i+2]==0):
                senal_Tx1[i*mpp : (i+1)*mpp] = portadora1*-np.sqrt(2)/2
                senal_Tx2[i*mpp : (i+1)*mpp] = portadora2*-np.sqrt(2)/2
            
            if(bits[i]==1 and bits[i+1]==0 and bits[i+2]==0):
                senal_Tx1[i*mpp : (i+1)*mpp] = portadora1*0
                senal_Tx2[i*mpp : (i+1)*mpp] = portadora2*-1
                
            if(bits[i]==1 and bits[i+1]==0 and bits[i+2]==1):
                senal_Tx1[i*mpp : (i+1)*mpp] = portadora1*np.sqrt(2)/2
                senal_Tx2[i*mpp : (i+1)*mpp] = portadora2*-np.sqrt(2)/2

    # Como son dos portadoras y como viajan juntas, se procede a sumar estos
    portadoraT = portadora1 + portadora2
    senal_Tx = senal_Tx1 +senal_Tx2
    
    # 5. Calcular la potencia promedio de la señal modulada
    P_senal_Tx = (1 / (N*Tc)) * np.trapz(pow(senal_Tx, 2), t_simulacion)
    
    # Retornar señales importantes
    return senal_Tx, P_senal_Tx, portadora1, portadora2 



def demodulador8(senal_Rx, portadora1, portadora2, mpp):
    '''Un método que simula un bloque demodulador
    de señales, bajo un esquema 8-PSK. El criterio
    de demodulación se basa en decodificación por 
    detección de energía.

    :param senal_Rx: La señal recibida del canal
    :param portadora: La onda portadora c(t)
    :param mpp: Número de muestras por periodo
    :return: Los bits de la señal demodulada
    '''
    # Cantidad de muestras en senal_Rx
    M = len(senal_Rx)
    
    # Cantidad de bits (símbolos) en transmisión
    N = int(M / mpp)
    
    # Vector para bits obtenidos por la demodulación
    bits_Rx = np.zeros(N)

    # Vector para la señal demodulada
    senal_demodulada1 = np.zeros(senal_Rx.shape)
    senal_demodulada2 = np.zeros(senal_Rx.shape)
    senal_demodulada = np.zeros(senal_Rx.shape)
        
    # Pseudo-energía de un período de la portadora
    Es1 = np.sum(portadora1 * portadora1)
    Es2 = np.sum(portadora2 * portadora2)

    # Demodulación
    for i in range(N):
        # Producto interno de dos funciones
        producto1 = senal_Rx[i*mpp : (i+1)*mpp] * portadora1
        Ep1 = np.sum(producto1) 
        senal_demodulada1[i*mpp : (i+1)*mpp] = producto1
        producto2 = senal_Rx[i*mpp : (i+1)*mpp] * portadora2
        Ep2 = np.sum(producto2) 
        senal_demodulada2[i*mpp : (i+1)*mpp] = producto2
        h = np.sqrt(2)/2
        senal_demodulada[i*mpp : (i+1)*mpp] = producto1+producto2 
        
        # Criterio de decisión por detección de energía
        if i % 3 == 0:
            if Ep1 >= (1+h)*Es1/2 and Ep2 <=h*Es2/2 and Ep2 >= -h*Es2/2:
                bits_Rx[i] = 1
                bits_Rx[i+1] = 1
                bits_Rx[i+2] = 1
            if Ep1 >= h*Es1/2 and Ep1 <= (1+h)*Es1/2 and Ep2 <= (h+1)*Es2/2 and Ep2 >= h*Es2/2:
                bits_Rx[i] = 1
                bits_Rx[i+1] = 1
                bits_Rx[i+2] = 0
            if Ep1 >= -h*Es1/2 and Ep1 <= h*Es1/2 and Ep2 >= (h+1)*Es2/2:
                bits_Rx[i] = 0
                bits_Rx[i+1] = 1
                bits_Rx[i+2] = 0
            if Ep1 >= -(h+1)*Es1/2 and Ep1 <= -h*Es1/2 and Ep2 <= (h+1)*Es2/2 and Ep2 >= h*Es2/2:
                bits_Rx[i] = 0
                bits_Rx[i+1] = 1
                bits_Rx[i+2] = 1
            if Ep1 <= -(h+1)*Es1/2 and Ep2 <=h*Es2/2 and Ep2 >= -h*Es2/2:
                bits_Rx[i] = 0
                bits_Rx[i+1] = 0
                bits_Rx[i+2] = 1
            if Ep1 >= -(h+1)*Es1/2 and Ep1 <= -h*Es1/2 and Ep2 <= -h*Es2/2 and Ep2 >= -(h+1)*Es2/2:
                bits_Rx[i] = 0
                bits_Rx[i+1] = 0
                bits_Rx[i+2] = 0
            if Ep1 >= -h*Es1/2 and Ep1 <= h*Es1/2 and Ep2 <= -(h+1)*Es2/2:
                bits_Rx[i] = 1
                bits_Rx[i+1] = 0
                bits_Rx[i+2] = 0
            if Ep1 >= h*Es1/2 and Ep1 <= (h+1)*Es1/2 and Ep2 <= -h*Es2/2 and Ep2 >= -(h+1)*Es2/2:
                bits_Rx[i] = 1
                bits_Rx[i+1] = 0
                bits_Rx[i+2] = 1
                
        else: 
            continue

    return bits_Rx.astype(int), senal_demodulada1, senal_demodulada2, senal_demodulada




import numpy as np
import matplotlib.pyplot as plt
import time

# Parámetros
fc = 5000  # frecuencia de la portadora (en comunicaciones inalambricas es 30000+)
mpp = 20   # muestras por periodo de la portadora
SNR = 20   # relación señal-a-ruido del canal

# Iniciar medición del tiempo de simulación
inicio1 = time.time()

# 1. Importar y convertir la imagen a trasmitir
imagen_Tx1 = fuente_info('arenal.jpg')
dimensiones1 = imagen_Tx1.shape

# 2. Codificar los pixeles de la imagen
bits_Tx1 = rgb_a_bit(imagen_Tx1)

# 3. Modular la cadena de bits usando el esquema 8-PSK
senal_Tx, Pm, portadora1, portadora2 = modulador8(bits_Tx1, fc, mpp)

# 4. Se transmite la señal modulada, por un canal ruidoso
senal_Rx = canal_ruidoso(senal_Tx, Pm, SNR)

# 5. Se desmodula la señal recibida del canal
bits_Rx8, senal_demodulada1, senal_demodulada2, senal_demodulada = demodulador8(senal_Rx, portadora1, portadora2, mpp)

# 6. Se visualiza la imagen recibida 
imagen_Rx8 = bits_a_rgb(bits_Rx8, dimensiones1)
Fig = plt.figure(figsize=(10,6))

# Cálculo del tiempo de simulación
print('Duración de la simulación: ', time.time() - inicio)

# 7. Calcular número de errores
errores = sum(abs(bits_Tx1 - bits_Rx8))
BER = errores/len(bits_Tx1)
print('{} errores, para un BER de {:0.4f}.'.format(errores, BER))

# Mostrar imagen transmitida
ax = Fig.add_subplot(1, 2, 1)
imgplot = plt.imshow(imagen_Tx1)
ax.set_title('Transmitido')

# Mostrar imagen recuperada
ax = Fig.add_subplot(1, 2, 2)
imgplot = plt.imshow(imagen_Rx8)
ax.set_title('Recuperado')
Fig.tight_layout()

plt.imshow(imagen_Rx8)




import matplotlib.pyplot as plt

# Visualizar el cambio entre las señales
fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True, figsize=(14, 7))

# La señal modulada por 8-PSK
ax1.plot(senal_Tx[0:600], color='g', lw=2) 
ax1.set_ylabel('$s(t)$')

# La señal modulada al dejar el canal
ax2.plot(senal_Rx[0:600], color='b', lw=2) 
ax2.set_ylabel('$s(t) + n(t)$')

# La señal demodulada
ax3.plot(senal_demodulada[0:600], color='m', lw=2) 
ax3.set_ylabel('$b^{\prime}(t)$')
ax3.set_xlabel('$t$ / milisegundos')
fig.tight_layout()
plt.show()




# Pruebas con otros SNR para observar la respuesta de la demodulación cuando la relación de ruido a señal
SNR = [-5, 0]

for i, snr in enumerate(SNR):
    import numpy as np
    import matplotlib.pyplot as plt
    import time

    # Parámetros
    fc = 5000  # frecuencia de la portadora (en comunicaciones inalambricas es 30000+)
    mpp = 20   # muestras por periodo de la portadora
    SNR = snr   # relación señal-a-ruido del canal

    # Iniciar medición del tiempo de simulación
    inicio1 = time.time()

    # 1. Importar y convertir la imagen a trasmitir
    imagen_Tx1 = fuente_info('arenal.jpg')
    dimensiones1 = imagen_Tx1.shape

    # 2. Codificar los pixeles de la imagen
    bits_Tx1 = rgb_a_bit(imagen_Tx1)

    # 3. Modular la cadena de bits usando el esquema 8-PSK
    senal_Tx, Pm, portadora1, portadora2 = modulador8(bits_Tx1, fc, mpp)

    # 4. Se transmite la señal modulada, por un canal ruidoso
    senal_Rx = canal_ruidoso(senal_Tx, Pm, SNR)

    # 5. Se desmodula la señal recibida del canal
    bits_Rx8, senal_demodulada1, senal_demodulada2, senal_demodulada = demodulador8(senal_Rx, portadora1, portadora2, mpp)

    # 6. Se visualiza la imagen recibida 
    imagen_Rx8 = bits_a_rgb(bits_Rx8, dimensiones1)
    Fig = plt.figure(figsize=(10,6))

    # Cálculo del tiempo de simulación
    print('Duración de la simulación: ', time.time() - inicio)

    # 7. Calcular número de errores
    errores = sum(abs(bits_Tx1 - bits_Rx8))
    BER = errores/len(bits_Tx1)
    print('{} errores, para un BER de {:0.4f}.'.format(errores, BER))

    # Mostrar imagen transmitida
    ax = Fig.add_subplot(1, 2, 1)
    imgplot = plt.imshow(imagen_Tx1)
    ax.set_title('Transmitido')

    # Mostrar imagen recuperada
    ax = Fig.add_subplot(1, 2, 2)
    imgplot = plt.imshow(imagen_Rx8)
    ax.set_title('Recuperado')
    Fig.tight_layout()

    plt.imshow(imagen_Rx8)




# Parte 2

# Tiempo final
T = 0.1

# Cantidad de tiempo
Tt = 100

# Se define el tiempo
t = np.linspace(0, T, Tt)

#Estacionaridad    
# Inicio de la funcion para N realizaciones
# Se define la variable h
h = np.sqrt(2)/ 2
A1 = [1, h, 0, -h, -1, -h, 0, h]
A2 = [0, h, 1, h, 0, -h, -1, -h]

# Se define N=8
N = 8      

# Se crean los vectores de las funciones 
Xt = np.empty((N, len(t)))   
xt = np.empty((N, len(t)))

# Se crean las funciones de las señales para el rango de amplitudes A
for i in range(N-1):
    
    # Se define X para cada i
    X = A1[i]*np.cos(2*np.pi*fc*t) + A2[i]*np.sin(2*np.pi*fc*t)
    
    # Se define x para cada i
    x = -A1[i]*np.cos(2*np.pi*fc*t) + A2[i]*np.sin(2*np.pi*fc*t)
    
    # Función de X(t) 
    Xt[i,:] = X
    
    # Función de x(t) 
    xt[i+1,:] = x
    
    # Gráfica de la señal X
    plt.plot(t, X)
    
    # Gráfica de la señal x
    plt.plot(t, x)

# Promedio para cada i
Prom = [np.mean(X)for i in range(len(t))]
Res = plt.plot(t, Prom, lw = 4, label = "Promedio de N realizaciones")
print("Promedio de N realizaciones para t: \n", Prom)

# Promedio teorico
Promteo = np.mean(senal_Tx)*t
print("El promedio del resultado teórico es: \n", Promteo)
Teo = plt.plot(t, Promteo, lw = 2.5, label="Resultado Teórico")

# Gráfica de realizaciones,promedio y promedios teóricos
plt.title('Resultados de realizaciones de $X(t)$')
plt.xlabel("Tiempo $(t)$")
plt.ylabel("Valor $x_i(t)$")
plt.legend()
plt.show()   

# Desplazamiento en Tt
desp = np.arange(Tt)

# Calculo de Tau
tau = desp/T

# Puntero para las correlaciones de las funciones
corr = np.empty((N, len(desp)))

# Figura de autocorrelación
plt.figure()

# Correlaciones de Tau
for n in range(N):
    for i, Tau in enumerate(desp):
        # Matriz de correlaciones
        corr[n, i] = np.correlate(Xt[n,:], np.roll(Xt[n,:], Tau))/T
    plt.plot(tau, corr[n,:])

# Calculo teórico de correlación 
p = 0.5
media = 0.3
Rxx = (p - np.power(media,2)) * np.cos(np.pi*tau)
print("La correlación es: \n", Rxx)

# Gráfica de correlaciones
plt.plot(tau, Rxx, lw = 2, label='Correlación teórica')
plt.title('Autocorrelaciones')
plt.xlabel("Tau")
plt.ylabel("Correlación $R_{xx}$($Tau$)")
plt.legend()
plt.show()

# Ergoicidad
maxi = max(senal_Tx)
mini = min(senal_Tx)

promtemp = (maxi+mini)/2

# Promedio estadistico
promesta = 0
for i in range(len(senal_Tx)):
    promesta += senal_Tx[i]

promesta = promesta/len(senal_Tx)

print("Promedios temporales: ")
print(promtemp)
print("Promedios estadisticos: ")
print(promesta)




# Parte 3
from scipy.fft import fft, ifft
import numpy as np
import matplotlib.pyplot as plt

# Transformada de Fourier
senal_f = fft(senal_Tx)

# Muestras de la señal
Nm = len(senal_Tx)

# Número de símbolos (198 x 89 x 8 x 3)
Ns = Nm // mpp

# Tiempo del símbolo = periodo de la onda portadora
Tc = 1 / fc

# Tiempo entre muestras (período de muestreo)
Tm = Tc / mpp

# Tiempo de la simulación
T = Ns * Tc

# Espacio de frecuencias
f = np.linspace(0.0, 1.0/(2.0*Tm), Nm//2)

# Densidad espectral
y = 2.0/Nm * np.power(np.abs(senal_f[0:Nm//2]), 2)

# Personalización del formato en la gráfica
fig, ax = plt.subplots(1, 1, figsize=(10, 5), dpi=100)
eje_x = "Frecuencia (Hz)"
eje_y = "Densidad espectral"
titulo = "Gráfica Densidad espectral de potencia"

# Gráficación de datos
ax.plot(f, y)
ax.set_xlabel(eje_x)
ax.set_ylabel(eje_y)
plt.suptitle(titulo)
plt.xlim(0, 20000)
plt.grid()
plt.show()


# ---
# 
# ### Universidad de Costa Rica
# #### Facultad de Ingeniería
# ##### Escuela de Ingeniería Eléctrica
# 
# ---
