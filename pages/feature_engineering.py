import streamlit as st
from PIL import Image
import pandas as pd
import time
import sys
import libs.feature_engineering as FE

def app():

    appSelectionSubCat = st.sidebar.selectbox('',['Home','Feature Combinations'])

    # Sub pages -------------------------------------------------------------------------------------------------

    def appFeatureCombination():

        st.header('Feature Combinations')
        
        optionDataset = st.selectbox(
        'Select the dataset',
        (st.session_state['dataset']))

        if optionDataset:

            with st.expander("Settings", expanded=False):

                varsBlackList = None
                varsWhiteList = None
                len_list_varsBlackList = 0
                len_list_varsWhiteList = 0
                n_cols = st.session_state[optionDataset].shape[1]


                # Black List -----------------------------------------------------------------------------------
                col1, col2 = st.columns([0.5,3])
                with col1:
                    has_bl = st.selectbox("Has blacklist", ['No', 'Yes'])
                with col2:
                    if has_bl == 'Yes':
                        varsBlackList = st.multiselect('Input the black list variables', list(st.session_state[optionDataset].columns))
                        len_list_varsBlackList = len(varsBlackList)

                # -----------------------------------------------------------------------------------

                # Black List -----------------------------------------------------------------------------------
                col1, col2 = st.columns([0.5,3])
                with col1:
                    has_wl = st.selectbox("Has whitelist", ['No', 'Yes'])
                with col2:
                    if has_wl == 'Yes':
                        varsWhiteList = st.multiselect('Input the white list variables', list(st.session_state[optionDataset].columns))
                        len_list_varsWhiteList = len(varsWhiteList)

                # -----------------------------------------------------------------------------------

                col1, col2, col3 = st.columns(3)
                with col1:
                    targetName = st.selectbox("Input the target variable", list(st.session_state[optionDataset].columns.insert(0,None)))
                    len_targetName = 0 if targetName == None else 1

                #value_max_coluns_combinations = n_cols - (len_list_varsBlackList + len_list_varsWhiteList) - len_targetName
                value_max_coluns_combinations = n_cols - len_list_varsBlackList - len_targetName

                with col2:
                    n_min = st.number_input('Input the minimum number of combinations', step=1, min_value=2)

                with col3:
                    n_max = st.number_input('Input the maximum number of combinations', step=1, value=value_max_coluns_combinations, max_value=value_max_coluns_combinations)

                col1, col2 = st.columns([1.53,3])
                with col1:
                    ObjectOptionsToSave = st.text_input("Input variable name")

                #click_teste = st.form_submit_button("Submit")
                click_teste = st.button("Submit")

                if click_teste:
                    if ObjectOptionsToSave:
                        if ObjectOptionsToSave not in st.session_state:
                            if targetName != None:

                                features = FE.featureCombinations(
                                        base=st.session_state[optionDataset],
                                        target=targetName,
                                        n_min = n_min,
                                        n_max = n_max,
                                        black_list = varsBlackList,
                                        white_list = varsWhiteList
                                )

                                st.session_state["Objects"].append(ObjectOptionsToSave)
                                st.session_state["Variables"].append(ObjectOptionsToSave)
                                st.session_state[ObjectOptionsToSave] = features.toCombine()
                                #st.markdown("**Combinations**")
                                #st.write(features.batch_combinations)
                                #st.markdown("**Total of combinations**")
                                #st.write(features.total_combination)

                                df_comb = pd.DataFrame(st.session_state[ObjectOptionsToSave])
                                df_comb.to_csv(f'files_output/combinations_{ObjectOptionsToSave}.csv', index=False)

                                st.success('Combination created successfully')
                                time.sleep(2)
                                st.experimental_rerun()

                            else:
                                st.error('You need put the target variable name')
                                time.sleep(0.1)
                                st.experimental_rerun()
                            
                        else:
                            st.error('This variable exists in the system')
                            time.sleep(0.1)
                            st.experimental_rerun()
                    else:
                        st.error('Input the variable name')
                        time.sleep(0.1)
                        st.experimental_rerun()

            with st.expander("Loading combinations", expanded=False):

                dataset_vars = list(st.session_state['dataset'])
                objects_vars = list(st.session_state['Objects'])
                objs_final = [objects_vars for objects_vars in objects_vars if objects_vars not in dataset_vars]

                NameVariable = st.selectbox(
                'Select the variable to save',
                (objs_final))

                if NameVariable:

                        myBase = st.radio("Select an option",('Upload csv','Upload xlsx'))
                        with st.form(key='my_form_load_combinations', clear_on_submit=True):
                            if not NameVariable in st.session_state['Variables']:
                                if myBase == 'Upload csv':

                                    check_delimiter = st.checkbox("Custom Delimiter")
                                    deLim = st.text_input('Input the delimiter')
                                    
                                    if check_delimiter:
                                        uploaded_file = st.file_uploader("Choose a file")
                                        if uploaded_file is not None:

                                            bytes_data = uploaded_file.getvalue()

                                            df_default = pd.read_csv(uploaded_file, sep=str(deLim))

                                            combs_list = []
                                            for i in range(0,len(df_default)):
                                                item = df_default.iloc[i,:].tolist()
                                                item = [item for item in item if str(item) != 'nan']
                                                combs_list.append(item)

                                            st.session_state["Variables"].append(NameVariable)
                                            st.session_state[NameVariable] = combs_list
                                            #st.session_state['have_dataset'] = True

                                    else:
                                        uploaded_file = st.file_uploader("Choose a file")
                                        if uploaded_file is not None:

                                            bytes_data = uploaded_file.getvalue()

                                            df_default = pd.read_csv(uploaded_file)

                                            combs_list = []
                                            for i in range(0,len(df_default)):
                                                item = df_default.iloc[i,:].tolist()
                                                item = [item for item in item if str(item) != 'nan']
                                                combs_list.append(item)

                                            st.session_state["Variables"].append(NameVariable)
                                            st.session_state[NameVariable] = combs_list
                                            #st.session_state['have_dataset'] = True

                                elif myBase == 'Upload xlsx':
                                        
                                    uploaded_file = st.file_uploader("Choose a file")
                                    if uploaded_file is not None:

                                        bytes_data = uploaded_file.getvalue()

                                        df_default = pd.read_excel(uploaded_file)

                                        combs_list = []
                                        for i in range(0,len(df_default)):
                                            item = df_default.iloc[i,:].tolist()
                                            item = [item for item in item if str(item) != 'nan']
                                            combs_list.append(item)

                                        st.session_state["Variables"].append(NameVariable)
                                        st.session_state[NameVariable] = combs_list
                                        #st.session_state['have_dataset'] = True

                            else:
                                st.write("<u><b>Variable loaded</b></u>", unsafe_allow_html=True)

                            submit_var_comb_load = st.form_submit_button('Submit')
                            if submit_var_comb_load:
                                #time.sleep(0.2)
                                st.experimental_rerun()

            with st.expander("View Variables", expanded=False):

                if st.session_state['Variables']:

                    SelectionViewVariable = st.selectbox(
                    'Select a variable', (st.session_state['Variables']))

                    # Contagem de combinações batch e total
                    lista_contagem = []
                    for x in range(0, len(st.session_state[SelectionViewVariable])):
                        lista_contagem.append(len(st.session_state[SelectionViewVariable][x]))
                        
                    total_combination = len(st.session_state[SelectionViewVariable])
                    batch_combinations = pd.DataFrame(lista_contagem, columns=['Qnt_var']).value_counts()
                    batch_combinations = pd.DataFrame(batch_combinations, columns=['Total'])

                    showbase = st.checkbox('Preview')
                    if showbase:
                        st.write("Total combinations", total_combination)
                        st.write("Features by combination", batch_combinations)
                        st.write("List of variables", st.session_state[SelectionViewVariable])
                else:
                    st.write("There is not variables loaded")


        else:

            st.write("There is no Dataset loaded")

    # -----------------------------------------------------------------------------------------------------------

    if appSelectionSubCat == 'Feature Combinations':
        appFeatureCombination()

    elif appSelectionSubCat == 'Home':

        st.image(Image.open('images/feature_engineering.png'), width=300)

        if st.session_state['have_dataset']:

            DatasetshowMeHome = st.selectbox(
            'Select a base', (st.session_state['dataset']))

        st.write(
        f"""

        Feature Engeneering
        ---------------
        - **There is dataset loaded?** {'Yes' if st.session_state.have_dataset else 'No'}
        - **Dataset rows**: {st.session_state[DatasetshowMeHome].shape[0] if st.session_state.have_dataset else None}
        - **Dataset columns**: {st.session_state[DatasetshowMeHome].shape[1] if st.session_state.have_dataset else None}
        """
        )

        if st.session_state['have_dataset']:
            def get_info(df):

                dfx_dtypes = pd.DataFrame(df.dtypes.values, columns=['type'])
                dfx_dtypes['type_str'] = dfx_dtypes['type'].apply(lambda x: str(x))
                x_dtypes = list(dfx_dtypes['type_str'].values)

                def sizeof_fmt(num):
                    #for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
                    for unit in ['','KByte','MByte','GByte','TByte','PByte','EByte','ZByte']:
                        if abs(num) < 1024.0:
                            return "%3.1f %s" % (num, unit)
                        num /= 1024.0
                    return "%.1f %s" % (num, 'Yi')

                # Salvando as colunas do dataframe em um dicionário
                dic_df = {}
                for key in df:
                    dic_df[key] = df[key]
                
                name_var_sizeof = []
                tam_var_sizeof = []
                df_sizeof = pd.DataFrame(columns=["Variable","Size"])

                for name, size in (((name, sys.getsizeof(value)) for name, value in dic_df.items())):
                    name_var_sizeof.append(name)
                    tam_var_sizeof.append(sizeof_fmt(size))
                    
                df_sizeof['Variable'] = name_var_sizeof
                df_sizeof['Size'] = tam_var_sizeof

                return pd.DataFrame({'Objects Types': x_dtypes, 'Size': df_sizeof['Size'].values, 'NaN': df.isna().sum(), 'NaN%': round((df.isna().sum()/len(df))*100,2), 'Unique':df.nunique()})

            st.markdown('**Variables**')
            st.dataframe(get_info(st.session_state[DatasetshowMeHome]))