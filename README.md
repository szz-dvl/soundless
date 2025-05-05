# EEG soundless app repository

Este repositorio contiene un conjunto de scripts escritos para trabajar con el dataset de [The Human Sleep Project](https://bdsp.io/content/hsp/2.0/).

## Scripts

Los scripts disponibles son los siguientes:

### parser_channels.py

Este script se encargará de descargar todas las configuraciones que existen en el dataset, los volcará en el fichero "out/channels.csv", este fichero lo usará el script channels.py para hacer un análisis de la frequencia de los canales y las configuraciones. Tiene un parámetro de configuración CHUNK_SIZE, este controlará cuantas entradas simultáneas se trataran, a mayor sea el entero configurado aquí, mayor velocidad obtendremos, pero más memória usaremos. El valor por defecto de este parámetro es 20.

### channels.py

Este script se encarga de, dada la salida del script anterior (parser_channels), generar un estudio de la frequencia de los canales y de las diferentes configuraciones en el dataset. Dejará los resultados en el fichero "out/channel_freqs.txt". Adicionalmente es script escribirá un fichero "out/equivalences.json" con las equivalencias hayadas en el proceso. El script tiene varios parámetros de configuración:

- CHANNEL_FREQUENCY_THRESHOLD: Este parámetro configurará que porcentage (entre 0 y 100) de apariciones en las diferentes configuraciones debe tener un canal para ser considerado relevante. Valor por defecto 70.

- CONF_GROUP_SIMILARITY_THRESHOLD: Para considerear dos configuraciones similares estas deben coincidir en un ratio mayor que el configurado en este parámetro. Valor por defecto 0.95.

- CHANNEL_EQUIVALENCE_SIMILARITY_THRESHOLD: Una vez obtenemos configuraciones de canales similares, desgranamos los canales de estas para obtener una diferencia de canales entre ellas, si esto canales son similares en un ratio mayor que el configurado en este parámetro, con considerados equivalentes. Valor por defecto 0.6.

- CONF_CONTRIBUTION_THRESHOLD: Para que una configuración sea considereada relevante, debe contribuir al dataset un ratio mayor que el establecido en este parámetro. Valor por defecto 0.05.

### parser_annotations.py

Este script generará un json que dejará en el fichero "out/annotations.json" con los resultados agregados de las diferentes anotaciones que encontramos en los ficheros de anotaciones de los encefalogramas, en particular la salida tendrá las ocurrencias totales de cada anotación, asi cómo las diferentes duraciones (en segundos) de cada una de las anotaciones (con las ocurrencias de cada duración), la diferencia entre la suma de las duraciones y las apariciones totales de cada anotación corresponden a valores no decodificables para las duraciones (NaN). Este script tiene un parámetro CHUNK_SIZE que de manera similar al script parser_channels configura la cantidad de entradas que se tratarán de manera simultánea, a mayor sea este entero, más rápido irá el script pero más memória usará. Valor por defecto 20.

### nn.py

Este script está en progreso en el momento de escribir esta documentación, pretende entrenar una red neuronal con las conclusiones sacadas de las salidas de los scripts anteriores, se descargará los encefalogramas, seleccionará una lista de 18 canales (que deben estar presentes) y las anotaciones asociadas con estos, iremos partiendo los encefalogramas en los trozos asociados con las anotaciones, y entrenaremos la red neuronal con los "trozos" de encefalograma y la anotacion pertinente cómo etiqueta asociada a la lectura. El parámetro habitual CHUNK_SIZE esta también disponible en este script, sin embargo se aconseja usarlo con cautela, ya que valores grandes pueden provocar que la memória del ordenador se agote dado el tamaño de los ficheros de los encefalogramas. Valor por defecto 2. Adicionalmente se ofrece el parámetro CHUNKS_TO_SAVE, que configura cada cuantos chunks volcamos el modelo a disco (10 por defecto).

### nn_single.py

Este script implementa una lógica muy parecida a nn.py, pero sin multithreading. Se ha implementado para mitigar problemas de falta de memória en el entrenamiento. Añade un parametro más CHUNKS_PER_TRAIN que configura cada cuantos chunks de datos entrenaremos la red neuronal, este se ha añadido para intentar mitigar el sesgo de los datos cuando estan asociados a un mismo paciente.

## Modulos

Los scripts anteriores dependen de una serie de modulos escritos para la ocasión. Estos estan localizados en el directório "modules" de este repositório. A groso modo, son los siguientes:

- aws: Se encarga de toda la comunicación con amazon, donde estan alojados los datos.
- chann_selector: La lógica de la selección de canales está implementada en este pequeño módulo, para hacerlo he usado las conclusiones que he sacado de la salida del script channels.py, de la que hay una cópia en "out/channel_freqs.txt".
- edf: Este modulo encapsúla la lógica asociada con los ficheros de los encefalogramas (.edf). Usa la librería de python [mne](https://mne.tools/stable/index.html) para facilitar el trabajo.
- model: Este modulo implementa la red neuronal con [keras](https://keras.io/).
- utils: Un pequeño modulo para implementar lógica que reuso en varios scripts.

## Entorno

Para usar los scripts es necesario tener en el mismo path donde esté este repositório el fichero "bdsp_psg_master_20231101.csv" que contiene la información de los encefalogramas, este se puede obtener pidiendo accesso a [The Human Sleep Project](https://bdsp.io/content/hsp/2.0/). Adicionalmente, también en el mismo directório dónde se encuentre este repositório, habrá que tener un fichero ".env" con las siguientes claves:

- AWS_KEY: Clave identificativa de un usuario de AWS acreditado por [The Human Sleep Project](https://bdsp.io/content/hsp/2.0/).
- AWS_SECRET_KEY: Secret key asociada a la clave anterior.
- AWS_BUCKET: Amazon bucket que contiene los datos, también facilitado por [The Human Sleep Project](https://bdsp.io/content/hsp/2.0/)
- AWS_REGION: Región que aloja los datos en Amazon.
- MODEL_CHECKPOINT_DIR: Directorio local donde se iran guardando los checkpoints de la red neuronal entrenada por el script nn.py.
