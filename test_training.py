"""
This is the Test  for Training the Machine Learning Model

"""


# Doing the necessary imports
from sklearn.model_selection import train_test_split
from data_ingestion import data_loader
from data_preprocessing import preprocessing
from data_preprocessing import clustering
from best_model_finder import tuner
from file_operations import file_methods
from application_logging import logger
import numpy as np
import pandas as pd

#Creating the common Logging object


path = 'Training_Batch_Files'
file_object = open("Training_Logs/ModelTrainingLog.txt", 'a+')
log_writer = logger.App_Logger()


try:
    # Getting the data from the source
    data_getter = data_loader.Data_Getter(file_object,log_writer)
    data = data_getter.get_data()
    print('###### Shape of original data ', data.shape)
    """Data Preprocessing"""

    preprocessor = preprocessing.Preprocessor(file_object,log_writer)
    
    # remove the column as it doesn't contribute to prediction.
    data.replace('?',np.NaN,inplace=True) # replacing '?' with NaN values for imputation
    
    # drop 14 columns 
    cols_to_drop = ['policy_number','policy_bind_date','policy_state','insured_zip','incident_location','incident_date','incident_state','incident_city','insured_hobbies','auto_make','auto_model','auto_year', 'age', 'total_claim_amount']
    data = preprocessor.remove_columns(data, cols_to_drop) 
    print('###### Shape of  data after irrelevant column removal', data.shape)

    # check if missing values are present in the dataset
    is_null_present, cols_with_missing_values = preprocessor.is_null_present(data)

    print('columns with missing values: ', cols_with_missing_values)
    # only handles  null in categorial data columns
    # if missing values are there, replace them appropriately.
    
    cat_df = data.select_dtypes(include=['object']).copy()
    num_df = data.select_dtypes(include=['int64','float64']).copy()

    print('###### Shape of categorial data and numerical data ', cat_df.shape, num_df.shape)

    data = preprocessor.encode_categorical_columns(data, is_null_present, cols_with_missing_values)

    print('###### Shape of  data after categorial data encoding', data.shape)

    # multiple times training with a same dataset causing duplicate record
    data.drop_duplicates(inplace = True)

    # create separate features and labels
    X,Y = preprocessor.separate_label_feature(data,label_column_name='fraud_reported')
    #Y = pd.DataFrame(Y, columns='frau_reported')
    #Y = Y.loc[:,~Y.columns.duplicated()]
    print('####### num of features in X', X.shape)
    print('####### num of features in Y', Y.shape)
    
    """ Applying the clustering approach"""

    kmeans = clustering.KMeansClustering(file_object,log_writer) # object initialization.
    number_of_clusters = kmeans.elbow_plot(X)  #  using the elbow plot to find the number of optimum clusters

    # Divide the data into clusters
    X = kmeans.create_clusters(X,number_of_clusters)

    #create a new column in the dataset consisting of the corresponding cluster assignments.
    X['Labels'] = Y['fraud_reported']
    
    # getting the unique clusters from our dataset
    list_of_clusters = X['Cluster'].unique()

    """parsing all the clusters and looking for the best ML algorithm to fit on individual cluster"""

    for i in list_of_clusters:
        cluster_data = X[X['Cluster']==i] # filter the data for one cluster

        # Prepare the feature and Label columns
        cluster_features = cluster_data.drop(['Labels','Cluster'],axis=1)
        cluster_label = cluster_data['Labels']

        print('splitting the data for cluster  ', i)
        # splitting the data into training and test set for each cluster one by one
        x_train, x_test, y_train, y_test = train_test_split(cluster_features, cluster_label, test_size=1 / 3, random_state=355)
        
        # Proceeding with more data pre-processing steps
        print('scaling numerical columns  data for cluster  ', i)

        x_train = preprocessor.scale_numerical_columns(x_train)
        x_test = preprocessor.scale_numerical_columns(x_test)

        print('finding model for cluster  ', i)

        model_finder = tuner.Model_Finder(file_object,log_writer) # object initialization

        #getting the best model for each of the clusters
        best_model_name,best_model=model_finder.get_best_model(x_train,y_train,x_test,y_test)

        #saving the best model to the directory.
        file_op = file_methods.File_Operation(file_object, log_writer)
        save_model = file_op.save_model(best_model,best_model_name+str(i))

except Exception as e:
    print("Error Occurred! %s" %e)