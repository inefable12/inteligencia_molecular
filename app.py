import streamlit as st
import pandas as pd
import joblib

from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen


# ============================================================
# CONFIGURACIÓN DE LA PÁGINA
# ============================================================

st.set_page_config(
    page_title="Clasificación molecular con K-Means",
    page_icon="🧪",
    layout="centered"
)


# ============================================================
# CARGAR MODELO
# ============================================================

@st.cache_resource
def cargar_modelo():
    return joblib.load("kmeans_model.pkl")


try:
    model = cargar_modelo()
except Exception as e:
    st.error("No se pudo cargar el modelo Kmeans_model.pkl")
    st.exception(e)
    st.stop()


# ============================================================
# INFORMACIÓN DE LOS CLUSTERS
# ============================================================

nombres_clusters = {
    0: "SIDERÓFORO",
    1: "FÁRMACO",
    2: "SOLVENTE"
}


# ============================================================
# DATOS DE EJEMPLO
# ============================================================

ejemplos = {
    "Etanol — CCO": {
        "smiles": "CCO",
        "h_implicitos": 6,
        "coef_de_part": -0.0014000000000000123,
        "MM": 46.069,
        "TPSA": 20.23,
        "cluster_esperado": 2
    },

    "Sideróforo — flavonoide": {
        "smiles": "C1=CC(=C(C=C1C2=C(C(=O)C3=C(C=C(C=C3O2)O)O)O)O)O",
        "h_implicitos": 20,
        "coef_de_part": -0.2445,
        "MM": 448.38,
        "TPSA": 190.28,
        "cluster_esperado": 0
    },

    "Fármaco — molécula compleja": {
        "smiles": r"CCCCCCCCCCCCCCCC(=O)OC[C@@H]\(CO[P@@]\(=O)([O-])O[C@@H]1[C@@H]\(O)[C@H]\(O)[C@@H]\(O)[C@H]\(O)[C@H]1O)OC(=O)CCCCCCC\C=C\CCCCCCCC",
        "h_implicitos": 79,
        "coef_de_part": 11.3399,
        "MM": 749.088,
        "TPSA": 136.0,
        "cluster_esperado": 1
    }
}


# ============================================================
# FUNCIONES PARA CALCULAR DESCRIPTORES
# ============================================================

def calcular_descriptores(smiles):
    """
    Calcula los cuatro descriptores utilizados por el modelo.
    """

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    # Número total de hidrógenos implícitos
    h_implicitos = sum(
        atomo.GetNumImplicitHs()
        for atomo in mol.GetAtoms()
    )

    # Coeficiente de partición
    coef_de_part = Crippen.MolLogP(mol)

    # Masa molecular
    MM = Descriptors.MolWt(mol)

    # Área superficial polar topológica
    TPSA = Descriptors.TPSA(mol)

    return {
        "h_implicitos": h_implicitos,
        "coef_de_part": coef_de_part,
        "MM": MM,
        "TPSA": TPSA
    }


def realizar_prediccion(datos):
    """
    Recibe un diccionario con los cuatro descriptores
    y realiza la predicción mediante K-Means.
    """

    nuevos_datos = pd.DataFrame({
        "h_implicitos": [datos["h_implicitos"]],
        "coef_de_part": [datos["coef_de_part"]],
        "MM": [datos["MM"]],
        "TPSA": [datos["TPSA"]]
    })

    prediccion = model.predict(nuevos_datos)

    cluster = int(prediccion[0])

    return cluster, nuevos_datos


# ============================================================
# TÍTULO
# ============================================================

st.title("🧪 Clasificación molecular con K-Means")

st.write(
    """
    Esta aplicación utiliza un modelo de **Machine Learning no supervisado
    (K-Means)** entrenado previamente para asignar una molécula a uno de
    tres grupos: **sideróforo, fármaco o solvente**.
    """
)

st.divider()


# ============================================================
# SELECCIÓN DEL MÉTODO DE INGRESO
# ============================================================

st.subheader("1. Seleccione los datos de entrada")

opcion = st.radio(
    "¿Cómo desea ingresar la molécula?",
    [
        "Seleccionar un ejemplo",
        "Ingresar los descriptores",
        "Ingresar SMILES"
    ]
)


# ============================================================
# OPCIÓN 1: EJEMPLOS
# ============================================================

if opcion == "Seleccionar un ejemplo":

    nombre_ejemplo = st.selectbox(
        "Seleccione una molécula:",
        list(ejemplos.keys())
    )

    ejemplo = ejemplos[nombre_ejemplo]

    st.markdown("### SMILES")

    st.code(
        ejemplo["smiles"],
        language="text"
    )

    datos = {
        "h_implicitos": ejemplo["h_implicitos"],
        "coef_de_part": ejemplo["coef_de_part"],
        "MM": ejemplo["MM"],
        "TPSA": ejemplo["TPSA"]
    }


# ============================================================
# OPCIÓN 2: INGRESAR DESCRIPTORES
# ============================================================

elif opcion == "Ingresar los descriptores":

    st.info(
        "Ingrese los cuatro descriptores utilizados durante "
        "el entrenamiento del modelo."
    )

    col1, col2 = st.columns(2)

    with col1:

        h_implicitos = st.number_input(
            "H implícitos",
            min_value=0,
            value=6,
            step=1
        )

        coef_de_part = st.number_input(
            "Coeficiente de partición (MolLogP)",
            value=-0.0014,
            format="%.6f"
        )

    with col2:

        MM = st.number_input(
            "Masa molecular (MM)",
            min_value=0.0,
            value=46.069,
            format="%.3f"
        )

        TPSA = st.number_input(
            "TPSA",
            min_value=0.0,
            value=20.23,
            format="%.3f"
        )

    datos = {
        "h_implicitos": h_implicitos,
        "coef_de_part": coef_de_part,
        "MM": MM,
        "TPSA": TPSA
    }


# ============================================================
# OPCIÓN 3: SMILES
# ============================================================

else:

    st.info(
        "Ingrese el código SMILES de una molécula. "
        "RDKit calculará automáticamente los cuatro descriptores."
    )

    smiles = st.text_input(
        "Código SMILES:",
        value="CCO",
        placeholder="Ejemplo: CCO"
    )

    if smiles.strip():

        datos_calculados = calcular_descriptores(smiles)

        if datos_calculados is None:

            st.error(
                "El SMILES ingresado no es válido. "
                "Verifique la estructura molecular."
            )

            st.stop()

        datos = datos_calculados

        st.markdown("### Descriptores calculados")

        tabla = pd.DataFrame({
            "Descriptor": [
                "H implícitos",
                "Coeficiente de partición (MolLogP)",
                "Masa molecular (MM)",
                "TPSA"
            ],
            "Valor": [
                datos["h_implicitos"],
                datos["coef_de_part"],
                datos["MM"],
                datos["TPSA"]
            ]
        })

        st.dataframe(
            tabla,
            use_container_width=True,
            hide_index=True
        )

    else:
        st.warning("Ingrese un código SMILES.")
        st.stop()


# ============================================================
# MOSTRAR DATOS UTILIZADOS POR EL MODELO
# ============================================================

st.divider()

st.subheader("2. Datos utilizados por K-Means")

datos_modelo = pd.DataFrame({
    "h_implicitos": [datos["h_implicitos"]],
    "coef_de_part": [datos["coef_de_part"]],
    "MM": [datos["MM"]],
    "TPSA": [datos["TPSA"]]
})

st.dataframe(
    datos_modelo,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# PREDICCIÓN
# ============================================================

st.divider()

if st.button(
    "🔬 Clasificar molécula",
    type="primary",
    use_container_width=True
):

    cluster, nuevos_datos = realizar_prediccion(datos)

    nombre_grupo = nombres_clusters.get(
        cluster,
        f"CLUSTER {cluster}"
    )

    st.subheader("3. Resultado")

    st.metric(
        "Cluster asignado",
        cluster
    )

    st.success(
        f"### {nombre_grupo}"
    )

    st.write(
        f"El modelo K-Means asignó la molécula al "
        f"**grupo {cluster}**, correspondiente a **{nombre_grupo}**."
    )

    # --------------------------------------------------------
    # Mostrar interpretación
    # --------------------------------------------------------

    if cluster == 0:

        st.info(
            "Según el modelo, las características moleculares "
            "de esta entrada corresponden al grupo de SIDERÓFOROS."
        )

    elif cluster == 1:

        st.info(
            "Según el modelo, las características moleculares "
            "de esta entrada corresponden al grupo de FÁRMACOS."
        )

    elif cluster == 2:

        st.info(
            "Según el modelo, las características moleculares "
            "de esta entrada corresponden al grupo de SOLVENTES."
        )

    # --------------------------------------------------------
    # Mostrar tabla final
    # --------------------------------------------------------

    st.markdown("### Vector utilizado para la predicción")

    st.code(
        nuevos_datos.to_string(index=False),
        language="text"
    )


# ============================================================
# INFORMACIÓN DEL MODELO
# ============================================================

with st.expander("ℹ️ Información sobre el modelo"):

    st.write(
        """
        **Algoritmo:** K-Means

        **Variables utilizadas:**

        - `h_implicitos`
        - `coef_de_part`
        - `MM`
        - `TPSA`

        **Interpretación de los clusters:**

        - Cluster 0 → SIDERÓFORO
        - Cluster 1 → FÁRMACO
        - Cluster 2 → SOLVENTE
        """
    )

    st.warning(
        "La interpretación de los clusters depende del modelo entrenado. "
        "Los números 0, 1 y 2 no tienen significado intrínseco en K-Means; "
        "aquí se han asociado a las categorías indicadas según su entrenamiento."
    )
