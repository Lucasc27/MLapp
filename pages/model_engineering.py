
import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import mlflow
import libs.EDA_graphs as EDA
import json
import time
import libs.model_engineering_new as me
import matplotlib.pyplot as plt
import seaborn as sns


def app():

    appSelectionSubCat = st.sidebar.selectbox('',['Home','Models Experiments', 'Report Experiments'])

    # Sub pages -------------------------------------------------------------------------------------------------

    def appModelsExperiments():

        st.header('Experiments')
        
        optionDataset = st.selectbox(
        'Select the dataset',
        (st.session_state['dataset']))

        if optionDataset:

            with st.expander("Settings", expanded=False):

                list_to_select_models = ["LogisticRegression", "XGBClassifier", "LGBMClassifier", "RandomForestClassifier", 
                "AdaBoostClassifier", "GradientBoostingClassifier", "LinearDiscriminantAnalysis", "GaussianNB", 
                "DecisionTreeClassifier", "KNeighborsClassifier"]
                n_rows_df = st.session_state[optionDataset].shape[0]

                nameExperiment = st.text_input("Input the name of experiment")

                # -----------------------------------------------------------------------------------
                col1, col2 = st.columns([0.9,3])
                with col1:
                    valid_features = st.selectbox("Combination or select features", ['None','Select features', 'Combination features'])
                with col2:
                    if valid_features == 'Select features':
                        combinations_features_pre = st.multiselect('Select the columns', st.session_state[optionDataset].columns)
                        combinations_features = []
                        combinations_features.append(list(combinations_features_pre))
                    elif valid_features == 'Combination features':
                        list_combination_features = st.selectbox('Select the variable', list(st.session_state["Variables"]))
                        combinations_features = st.session_state[list_combination_features]

                # -----------------------------------------------------------------------------------
                col1, col2, col3 = st.columns([1,1,1])
                with col1:
                    targetName = st.selectbox("Input the target name", list(st.session_state[optionDataset].columns.insert(0,'None')))
                with col2:
                    split_dataset = st.number_input("Input the sample size (%)", step=0.01, value=1.0, min_value=0.0, max_value=1.0)
                with col3:
                    run_test_size = st.number_input("Input the test sample size (%)", step=0.01, value=0.3, min_value=0.01, max_value=0.9)

                # -----------------------------------------------------------------------------------
                col1, col2 = st.columns([0.5,3])
                with col1:
                    select_or_input_models = st.selectbox("Models", ['None','Select models', 'Input models'])
                with col2:
                    if select_or_input_models == 'Select models':
                        list_models_runs = st.multiselect('Select the models', list_to_select_models)
                    elif select_or_input_models == 'Input models':
                        st.error("construction....")
                # -----------------------------------------------------------------------------------

                standardScaler_data = st.checkbox("StandardScale")
                balancing_data = st.checkbox("Balance data (SMOTE)")
                dummies = st.checkbox("Dummy transform")
                shap_output = st.checkbox("Shap output")

                click_run = st.button("Submit")

                if click_run:

                    if valid_features == 'None':
                        combinations_features = None

                    if not nameExperiment:
                        nameExperiment = 'MLFLOW MODELS TESTING'

                    if split_dataset == 1.0:
                        split_dataset = None

                    run_models = me.modelSetting(
                        base = st.session_state[optionDataset], target = targetName,
                        split_sample = split_dataset, cols_combinations = combinations_features,
                        test_size = run_test_size, models = list_models_runs, mlflow_name_experiment = nameExperiment)

                    run_models.execute()

        
        else:

            st.write("There is no Dataset loaded")

    def appReportExperiments():

        st.header('Report Experiments')

        experiments = mlflow.list_experiments()
        list_name_of_experiments = [experiments.name for experiments in experiments]
        list_name_of_experiments.insert(0, 'None')

        optionExperimentsName = st.selectbox(
        'Select the experiment',
        (list_name_of_experiments))

        if experiments:
            if optionExperimentsName != 'None':

                NameExperiment = optionExperimentsName
                experiment_id = mlflow.get_experiment_by_name(NameExperiment).experiment_id

                st.write(
                f"""

                Infos:
                ---------------
                - **Experiment name**: {NameExperiment}
                - **Experiment id**: {experiment_id}
                - **Operations totals**: {len(mlflow.list_run_infos(experiment_id))}
                - **Completed operations**: {len([x for x in mlflow.list_run_infos(experiment_id) if x.status == 'FINISHED'])}
                - **Failed operations**: {len([x for x in mlflow.list_run_infos(experiment_id) if x.status == 'FAILED'])}
                """
                )

                with st.expander("List of runs", expanded=False):

                    list_ids_runs =  [x.run_id for x in mlflow.list_run_infos(experiment_id) if x.status == 'FINISHED']

                    df_runs = pd.DataFrame()
                    for id_run in list_ids_runs:
                        run = mlflow.get_run(id_run)
                        with open(f'mlruns/{experiment_id}/{id_run}/artifacts/columns.txt') as f:
                            json_data = json.load(f)
                        
                        dic_metrics = run.data.metrics
                        dic_metrics['time_end'] = round((run.info.end_time - run.info.start_time) / 1000,2)
                        dic_metrics['Algorithm'] = run.data.tags['mlflow.runName']
                        dic_metrics['id_run'] = id_run
                        dic_metrics['Columns'] = json_data['columns']
                        dic_metrics['Target'] = json_data['target']
                        
                        infos_runs = pd.DataFrame([dic_metrics])
                        df_runs = pd.concat([df_runs, infos_runs], axis=0)
                    df_runs.reset_index(drop=True, inplace=True)

                    cols_to_filter = df_runs.drop(['X train','X test','Algorithm','id_run','Columns','Target'],axis=1).columns
                    col1, col2 = st.columns([0.5,2])
                    with col1:
                        order = st.selectbox("Ascending", [False,True])
                    with col2:
                        orderbyDF = st.multiselect("Order By", cols_to_filter)

                    # -----------------------------------------------------------------------------------------------
                    cols_to_filter_2 = df_runs.drop(['Algorithm','Columns','Target'],axis=1).columns
                    col1, col2, col3 = st.columns([1,1,1])
                    with col1:
                        filterbyDF_variable = st.selectbox("Select the variable", cols_to_filter_2)
                    with col2:
                        filterbyDF_controller = st.selectbox("", ["<", "=", ">", "=="])
                    with col3:
                        if filterbyDF_controller == '==':
                            filterbyDF_value_compare = st.text_input("Input")
                        else:
                            filterbyDF_value = st.number_input("Value", min_value=0, value=0, step=1)

                    col1, col2, col3 = st.columns([0.5,0.5,3.8])
                    with col1:
                        submit_filter = st.button("Submit filter")
                    with col2:
                        submit_filter_reset = st.button("Reset filter")
                    #------------------------------------------------------------------------------------------------

                    df_runs_filter = df_runs.sort_values(by=orderbyDF, ascending=order)

                    if submit_filter:
                        if filterbyDF_controller == '<':
                            df_runs_filter = df_runs_filter[df_runs_filter[filterbyDF_variable] < filterbyDF_value]
                        elif filterbyDF_controller == '=':
                            df_runs_filter = df_runs_filter[df_runs_filter[filterbyDF_variable] == filterbyDF_value]
                        elif filterbyDF_controller == '>':
                            df_runs_filter = df_runs_filter[df_runs_filter[filterbyDF_variable] > filterbyDF_value]

                    if submit_filter_reset:
                        st.experimental_rerun()

                    #cm = sns.light_palette("green", as_cmap=True)
                    cm = sns.color_palette("Blues", as_cmap=True)
                    #cm = sns.dark_palette("#69d", reverse=False, as_cmap=True)
                    st.write(df_runs_filter.style.background_gradient(cmap=cm))

                    st.write("-------------------------------------")
                    filter_to_barplot = st.selectbox("Select the variable", cols_to_filter.insert(0, 'None'))
                    if filter_to_barplot != 'None':
                        fig, ax = plt.subplots()
                        #plt.figure(figsize=(20, 10))
                        sns.barplot(x=filter_to_barplot, y="id_run", data=df_runs.sort_values(by=filter_to_barplot, ascending=False))
                        #plt.title('Important Features')
                        #plt.tight_layout()
                        #plt.show()
                        #obj_plot.CountPlot(col_count_plot, hue_opt)
                        st.pyplot(fig)
                    else:
                        st.write("Select the variable to plotting")
                
                with st.expander("Charts", expanded=False):

                    eda_plot = EDA.EDA(df_runs)

                    def plot_multivariate(obj_plot, radio_plot):
                        
                        if 'Boxplot' in radio_plot:
                            st.subheader('Boxplot')
                            col_y  = st.multiselect("Choose main variable (numerical)",obj_plot.num_vars, key ='boxplot_multivariate')
                            #col_x  = st.selectbox("Choose x variable (categorical) *optional", obj_plot.columns.insert(0,None), key ='boxplot_multivariate')
                            hue_opt = st.selectbox("Hue (categorical) *optional", obj_plot.columns.insert(0,None), key ='boxplot_multivariate')
                            #if st.sidebar.button('Plot boxplot chart'):
                            st.plotly_chart(obj_plot.box_plot(col_y, hue_opt))
                        
                        if 'Histogram' in radio_plot:
                            st.subheader('Histogram')
                            col_hist = st.selectbox("Choose main variable", obj_plot.num_vars, key = 'hist')
                            hue_opt = st.selectbox("Hue (categorical) optional",obj_plot.columns.insert(0,None), key = 'hist')
                            bins_, range_ = None, None
                            bins_ = st.slider('Number of bins optional', value = 30)
                            range_ = st.slider('Choose range optional', int(obj_plot.df[col_hist].min()), int(obj_plot.df[col_hist].max()),\
                                    (int(obj_plot.df[col_hist].min()),int(obj_plot.df[col_hist].max())))    
                            #if st.button('Plot histogram chart'):
                            st.plotly_chart(obj_plot.histogram_num(col_hist, hue_opt, bins_, range_))

                        if 'Scatterplot' in radio_plot: 
                            st.subheader('Scatter plot')
                            col_x = st.selectbox("Choose X variable (numerical)", obj_plot.num_vars, key = 'scatter')
                            col_y = st.selectbox("Choose Y variable (numerical)", obj_plot.num_vars, key = 'scatter')
                            hue_opt = st.selectbox("Hue (categorical) optional", obj_plot.columns.insert(0,None), key = 'scatter')
                            size_opt = st.selectbox("Size (numerical) optional",obj_plot.columns.insert(0,None), key = 'scatter')
                            #if st.sidebar.button('Plot scatter chart'):
                            st.plotly_chart(obj_plot.scatter_plot(col_x,col_y, hue_opt, size_opt))

                        if 'Countplot' in radio_plot:
                            st.subheader('Count Plot')
                            col_count_plot = st.selectbox("Choose main variable",obj_plot.columns, key = 'countplot')
                            hue_opt = st.selectbox("Hue (categorical) optional",obj_plot.columns.insert(0,None), key = 'countplot')
                            #if st.sidebar.button('Plot Countplot'):
                            fig, ax = plt.subplots()
                            obj_plot.CountPlot(col_count_plot, hue_opt)
                            st.pyplot(fig)

                    radio_plot = st.multiselect('Choose plot style', ['Boxplot', 'Histogram', 'Scatterplot', 'Countplot'])

                    plot_multivariate(eda_plot, radio_plot)

            else:
                st.write("Select the experiment!")
        else:
            st.write("There is no experiments loaded")


    # -----------------------------------------------------------------------------------------------------------

    if appSelectionSubCat == 'Models Experiments':
        appModelsExperiments()

    elif appSelectionSubCat == 'Report Experiments':
        appReportExperiments()

    elif appSelectionSubCat == 'Home':

        st.image(Image.open('images/image7.png'), width=300)

        if st.session_state['have_dataset']:

            DatasetshowMeHome = st.selectbox(
            'Select a base', (st.session_state['dataset']))

        st.write(
        f"""

        Model Engeneering
        ---------------
        - **There is dataset loaded?** {'Yes' if st.session_state.have_dataset else 'No'}
        - **Dataset rows**: {st.session_state[DatasetshowMeHome].shape[0] if st.session_state.have_dataset else None}
        - **Dataset columns**: {st.session_state[DatasetshowMeHome].shape[1] if st.session_state.have_dataset else None}
        """
        )