import numpy
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def preprocessFeaturesSvr( featuresTrain, featuresTest = None ):
    scaler = StandardScaler().fit( featuresTrain )
    if featuresTest is None:
        return scaler.transform( featuresTrain )
    else:
        return scaler.transform( featuresTrain ), scaler.transform( featuresTest )
