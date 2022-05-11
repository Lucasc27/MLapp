
import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
#from sympy import python
import libs.EDA_graphs as EDA
#import time
#import libs.model_engineering_new as me
import matplotlib.pyplot as plt
import seaborn as sns


def app():

    appSelectionSubCat = st.sidebar.selectbox('',['Home','Automated Pre-processing','Execute Script .py'])

    # Sub pages -------------------------------------------------------------------------------------------------

    def appAutomatedPreProcessing():

        st.header('Automated Pre-Processing')

        with st.expander("Click here for more info on this app section", expanded=False):

            st.write(
            f"""

            Summary
            ---------------
            Welcome to the application pre-processing session. Pre-processing is a very important step in AI model 
            development or data analysis. This app provides a quick way to use preprocessing methods on your dataset.

            **select the dataset and proceed with the execution**
            """
            )
        
        optionDataset = st.selectbox(
        'Select the dataset',
        (st.session_state['dataset']))

        if optionDataset:

            #df = st.session_state[optionDataset]
            dataset_columns = st.session_state[optionDataset].columns
            options_columns = dataset_columns.insert(0, 'None')

            # ----------------------------- AQUI

            st.header("Step 1")
            st.subheader("Type of learning")

            type_model = st.selectbox('Select the type of model you are preparing data for:',
            ('None', 'Unsupervised', 'Supervised'))

            if type_model == 'Supervised':

                what_target_variable = st.selectbox("Select target variable", options_columns)

                if what_target_variable != 'None':
                    st.write("-----------------------------------")
                    st.header("Step 2")
                    st.subheader("Remove special characters")

                    col1_remov_caract, col2_remov_caract, col3_remov_caract = st.columns([1,2,3])
                    with col1_remov_caract:
                        will_remove_caracters = st.selectbox("Remove characters?", ['None','Yes', 'No'])

                    if will_remove_caracters != 'None':
                        st.write("-----------------------------------")
                        st.header("Step 3")
                        st.subheader("Remove variables")

                        var_remove_pass = False
                        col1_remov_var, col2_remov_var, col3_remov_var, col4_remov_var = st.columns([1,1,1.5,2.5])
                        with col1_remov_var:
                            will_remove_var = st.selectbox("Remove variables?", ['None','Yes', 'No'])
                        with col2_remov_var:
                            if will_remove_var == 'Yes':
                                will_remove_select_or_input = st.selectbox("Select or insert variables?",['None','Select','Insert'])
                            elif will_remove_var == 'No':
                                var_remove_pass = True
                        with col3_remov_var:
                            if will_remove_var == 'Yes' and will_remove_select_or_input == 'Select':
                                var_remove = st.multiselect("Select the variables to remove", st.session_state[optionDataset].columns)
                                if var_remove:
                                    var_remove_pass = True
                            elif will_remove_var == 'Yes' and will_remove_select_or_input == 'Insert':
                                var_remove = st.text_input("Select variables to remove separated by commas")
                                var_remove = var_remove.split(',')
                                if len(var_remove) > 0 and var_remove[0] != '':
                                    var_remove_pass = True

                        if will_remove_var != 'None' and var_remove_pass:
                            st.write("-----------------------------------")
                            st.header("Step 4")
                            st.subheader("Filter dataset")

                            var_filter_pass = False
                            col1_filtering, col2_filtering, col3_filtering = st.columns([1,1,3])
                            with col1_filtering:
                                will_filter = st.selectbox("Filter dataset?", ['None','Yes', 'No'])
                            
                            if will_filter == "Yes":
                                col1_filter_manys, col2_filter_manys, col3_filter_manys, col4_filter_manys, col5_filter_manys  = st.columns([0.8,0.4,0.4,0.8,2])
                                with col1_filter_manys:
                                    column_to_filter = st.selectbox("Select the column", st.session_state[optionDataset].columns)
                                with col2_filter_manys:
                                    comparation = st.selectbox("Conditional", ['>','>=','<','<=','==','!='])
                                with col3_filter_manys:
                                    type_value_to_filter = st.selectbox("Type filter", ["Numerical", "Categorical"])
                                with col4_filter_manys:
                                    if type_value_to_filter == 'Numerical':
                                        value_to_filter = st.text_input("Filter by")
                                    elif type_value_to_filter == 'Categorical':
                                        value_to_filter = st.number_input("Filter by", value=1234567890)
                                code = f'''dataset[dataset[{column_to_filter}] {comparation} {value_to_filter}]'''
                                st.code(code, language='python')
                                if type_value_to_filter == 'Numerical':
                                    if column_to_filter and comparation and value_to_filter != 1234567890:
                                        var_filter_pass = True
                                elif type_value_to_filter == 'Categorical':
                                    if column_to_filter and comparation and value_to_filter != '':
                                        var_filter_pass = True

                            elif will_filter == "No":
                                var_filter_pass = True

                            if will_filter != 'None' and var_filter_pass:
                                st.write("-----------------------------------")
                                st.header("Step 5")
                                st.subheader("Submited")

                                submit_transformations = st.button("Apply")

                                if submit_transformations:

                                    # Removendo as colunas
                                    if will_remove_var == "Yes" and var_remove_pass and var_remove[0] != '':

                                        st.session_state[optionDataset].drop(var_remove,axis=1, inplace=True)

                                    # Limpando os caracteres especiais
                                    if will_remove_caracters == "Yes":

                                        df_cols = pd.DataFrame(st.session_state[optionDataset].columns, columns=['cols'])
                                        df_cols['cols'] = df_cols['cols'].replace('/', '_', regex=True).replace('\W+', '', regex=True).replace('__', '', regex=True).str.upper()
                                        st.session_state[optionDataset].columns = df_cols['cols'].values

                                        for c in st.session_state[optionDataset].select_dtypes(include=['object','category']).columns:
                                            st.session_state[optionDataset][c] = st.session_state[optionDataset][c].replace('/', '_', regex=True).replace('\W+', '', regex=True).replace('__', '', regex=True).str.lower()

                                    # Filtrando o conjunto de dados
                                    if will_filter == "Yes" and var_filter_pass:
                                        dataset_filter = st.session_state[optionDataset] 
                                        
                                        if comparation == '>':
                                            st.session_state[optionDataset][st.session_state[optionDataset][column_to_filter] > value_to_filter]
                                        elif comparation == '>=':
                                            st.session_state[optionDataset][st.session_state[optionDataset][column_to_filter] >= value_to_filter]
                                        elif comparation == '<':
                                            st.session_state[optionDataset][st.session_state[optionDataset][column_to_filter] < value_to_filter]
                                        elif comparation == '<=':
                                            st.session_state[optionDataset][st.session_state[optionDataset][column_to_filter] <= value_to_filter]
                                        elif comparation == '==':
                                            st.session_state[optionDataset][st.session_state[optionDataset][column_to_filter] == value_to_filter]
                                        elif comparation == '!=':
                                            st.session_state[optionDataset][st.session_state[optionDataset][column_to_filter] != value_to_filter]

                                        


            elif type_model == 'Unsupervised':

                st.error("Coming soon...")

            st.write("-------------------------")

            with st.expander("View dataset information",expanded=False):

                view_dataset = st.checkbox("View dataset")
                if view_dataset:
                    if list(st.session_state[optionDataset].select_dtypes(include=[np.number]).columns):
                        tab_numbers = st.session_state[optionDataset].describe(include=[np.number]).T
                    else:
                        tab_numbers = pd.DataFrame()
                    if list(st.session_state[optionDataset].select_dtypes(include=['object','category']).columns):
                        tab_objects = st.session_state[optionDataset].describe(include=['object','category']).T
                    else:
                        tab_objects = pd.DataFrame()

                    tab_joins = pd.concat([tab_objects, tab_numbers],axis=1)

                    tab_nans = pd.DataFrame(st.session_state[optionDataset].isna().sum(), columns=['NaN'])
                    tab_nans['NaN%'] = round((st.session_state[optionDataset].isna().sum() / st.session_state[optionDataset].shape[0])*100,2)

                    tab = tab_joins.join(tab_nans)

                    dfx_dtypes = pd.DataFrame(st.session_state[optionDataset].dtypes, columns=['type'])
                    dfx_dtypes['type_str'] = dfx_dtypes['type'].apply(lambda x: str(x))
                    del dfx_dtypes['type']

                    tab_final = tab.join(dfx_dtypes)
                    if tab_objects.shape[0] > 0 and tab_numbers.shape[0] > 0:
                        tab_final.columns = ['count', 'unique', 'top', 'freq', 'count_number', 'mean', 'std', 
                            'min', '25%','50%', '75%', 'max', 'NaN', 'NaN%', 'type_str']

                        tab_final['count'] = np.where(tab_final['count'].isnull(), tab_final['count_number'], tab_final['count'])
                        tab_final['count'] = tab_final['count'].astype(int)
                        del tab_final['count_number']

                    col1, col2, col3 = st.columns([0.8,0.8,2.5])
                    with col1:
                        init_rows = st.number_input("Min. of rows", value=0, min_value=0)
                    with col2:
                        final_rows = st.number_input("Max. of rows", value=5, min_value=0, max_value=int(st.session_state[optionDataset].shape[0]))
                        
                    col11, col22, col33 = st.columns([0.8,0.8,2.5])
                    with col11:
                        init_cols = st.number_input("Min. of cols", value=0, min_value=0)
                    with col22:
                        final_cols = st.number_input("Max. of cols", value=int(st.session_state[optionDataset].shape[1]), min_value=0, max_value=int(st.session_state[optionDataset].shape[1]))

                    selectionColumn = st.checkbox('Columns selections')

                    st.header("View dataset")
                    if selectionColumn:
                        optionsViewColumns = st.multiselect("Multiselect columns",list(st.session_state[optionDataset].columns))
                        #st.write(optionsViewColumns)
                        st.write(
                            f"""
                            **Columns selected**: {", ".join(optionsViewColumns)}
                            """
                        )
                        st.dataframe(st.session_state[optionDataset][optionsViewColumns].iloc[init_rows:final_rows])
                    else:
                        st.dataframe(st.session_state[optionDataset].iloc[init_rows:final_rows, init_cols:final_cols])

                    st.write("------------------------------------------------------")
                    st.header("Descriptive")
                    st.dataframe(tab_final)

        else:

            st.write("There is no Dataset loaded")
            

    def appExecuteScript():

        st.header('Execute script .py')

        with st.expander("Click here for more info on this app section", expanded=False):

            st.write(
            f"""

            Summary
            ---------------
            Welcome to the application's scripting session. So that you can have more options for processing your dataset,
            you can run functions directly from a python script.

            **select the dataset and proceed with the execution**
            """
            )
        
        optionDataset = st.selectbox(
        'Select the dataset',
        (st.session_state['dataset']))

        if optionDataset:

            uploaded_script = st.file_uploader("Choose a file", accept_multiple_files=False)
            name_script = uploaded_script.name
            st.write(name_script)

        else:

            st.write("There is no Dataset loaded")

    # -----------------------------------------------------------------------------------------------------------


    if appSelectionSubCat == 'Automated Pre-processing':
        appAutomatedPreProcessing()

    elif appSelectionSubCat == 'Execute Script .py':
        appExecuteScript()

    elif appSelectionSubCat == 'Home':

        st.image(Image.open('images/image8.png'), width=300)

        if st.session_state['have_dataset']:

            DatasetshowMeHome = st.selectbox(
            'Select a base', (st.session_state['dataset']))

        st.write(
        f"""

        Data Preparation
        ---------------
        - **There is dataset loaded?** {'Yes' if st.session_state.have_dataset else 'No'}
        - **Dataset rows**: {st.session_state[DatasetshowMeHome].shape[0] if st.session_state.have_dataset else None}
        - **Dataset columns**: {st.session_state[DatasetshowMeHome].shape[1] if st.session_state.have_dataset else None}
        """
        )



