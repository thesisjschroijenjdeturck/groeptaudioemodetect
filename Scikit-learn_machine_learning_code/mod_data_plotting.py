import numpy as np

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

import emo_detect_auto_ml_global_consts
import mod_data_loading as data_loading
import mod_pca as pca
    
def plotFeaturesOverLabels9x():
    fig = plt.figure(figsize=(16,12))
    for index, curSeed in enumerate([ 38, 39, 40, 41, 42, 43, 44, 45, 46 ]):
        features, arousalLabels, valenceLabels = data_loading.loadData( 'DEVEL', 1, emo_detect_auto_ml_global_consts.develSetSize, 0.01, curSeed )
        featuresPrincipalComponents = pca.transformToPrincipalComponents( features )   
        featuresPrincipalComponents = np.linalg.norm( featuresPrincipalComponents, axis=1 )
        plotIndex = index + 1
        ax = fig.add_subplot( ( 330 + plotIndex ), projection='3d' )
        ax.scatter( arousalLabels, valenceLabels, np.zeros(len(arousalLabels)), c='b', marker='o', label='Labels' )
        ax.scatter( arousalLabels, valenceLabels, featuresPrincipalComponents, c='r', marker='o', label='Labels > features' )
        ax.set_xlabel( 'Arousal' )
        ax.set_ylabel( 'Valence' )
        ax.set_zlabel( '|Feature PCA\'s|' )
        ax.legend()
    fig.savefig('featuresOverLabels.png')
    plt.show()