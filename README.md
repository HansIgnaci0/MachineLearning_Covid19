Proyecto de Clustering No Supervisado con Pipelines Automatizados

Este documento describe un flujo completo de trabajo que incluye: -
Preprocesamiento de datos - Entrenamiento de modelos no supervisados -
Generación de visualizaciones - Exportación de resultados - Orquestación
automática mediante Airflow (sin incluir credenciales)

1. Modelos No Supervisados Utilizados

Se implementaron tres algoritmos clásicos y fáciles de configurar:

1. K-Means

-   Algoritmo particional basado en centroides.
-   Rápido, simple y eficiente.
-   Ideal para datos que forman clusters relativamente esféricos.

2. Gaussian Mixture Models (GMM)

-   Clustering probabilístico.
-   Permite clusters elípticos.
-   Produce etiquetas similares a K-Means cuando los datos tienen poca
    variación estructural.

3. Hierarchical Clustering

-   Construye una estructura jerárquica (dendrograma).
-   Útil cuando se desea entender relaciones entre puntos.
-   Configuración sencilla usando AgglomerativeClustering.

------------------------------------------------------------------------

2. CSV con Resultados

El CSV generado automáticamente debe incluir:

    id, feature_1, feature_2, ..., cluster

Recomendaciones: - Agregar un ID incremental para identificar cada
fila. - Agregar columnas originales o procesadas antes de clusterizar. -
El campo cluster mostrará la etiqueta asignada (0,1,2…).

------------------------------------------------------------------------

3. Por Qué los Gráficos Pueden Verse Idénticos

Los tres modelos pueden generar visualizaciones muy similares cuando: -
Los datos tienen solo 1 o 2 clusters claros. - Las variables no muestran
estructuras complejas. - K-Means, GMM y Hierarchical convergen a las
mismas separaciones.

Esto es normal y no significa que esté mal implementado.

------------------------------------------------------------------------

4. Flujo Automático (Pipeline)

El sistema realiza:

1.  Carga de datos (load_data)
2.  Preprocesamiento (clean_data)
3.  Generación de modelos (train_models)
4.  Exportación (save_results)
5.  Visualización (plot_clusters)

Toda la ejecución está orquestada en Airflow sin exponer credenciales.

------------------------------------------------------------------------

5. DAG de Airflow (sin credenciales)

    from airflow import DAG
    from airflow.operators.python import PythonOperator
    from datetime import datetime
    import pandas as pd

    def load_data():
        pass  # Tu lógica aquí

    def preprocess():
        pass

    def clustering():
        pass

    def save_outputs():
        pass

    with DAG(
        dag_id="pipeline_clustering",
        start_date=datetime(2025, 1, 1),
        schedule_interval="@daily",
        catchup=False
    ):
        t1 = PythonOperator(task_id="load_data", python_callable=load_data)
        t2 = PythonOperator(task_id="preprocess", python_callable=preprocess)
        t3 = PythonOperator(task_id="clustering", python_callable=clustering)
        t4 = PythonOperator(task_id="save_outputs", python_callable=save_outputs)

        t1 >> t2 >> t3 >> t4

------------------------------------------------------------------------

6. Consideraciones Finales

-   Los modelos no supervisados no requieren una variable objetivo, por
    eso se llaman “no supervisados”.
-   El objetivo de la evaluación es demostrar correcto uso de:
    -   Técnicas de clustering
    -   Gráficos comparativos
    -   Automatización del flujo
-   Si los clusters se ven iguales, probablemente es resultado natural
    de los datos.

------------------------------------------------------------------------

7. Entregables Sugeridos

-   Informe PDF o Markdown
-   Código en Jupyter o Python scripts
-   CSV final con etiquetas
-   Gráficos PNG/JPG
-   DAG de Airflow
