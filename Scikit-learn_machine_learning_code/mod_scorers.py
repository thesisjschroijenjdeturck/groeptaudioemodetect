import numpy

from sklearn.metrics.scorer import make_scorer 
from tensorflow import convert_to_tensor

module_tag = "MOD_SCORERS"

def calculateCcc( x, y ):
    xMean = numpy.nanmean( x )
    yMean = numpy.nanmean( y )

    covariance = numpy.nanmean( ( x - xMean ) * ( y - yMean ) )
    
    xVariance = ( 1.0 / ( len( x ) - 1 ) ) * numpy.nansum( ( x - xMean ) ** 2 )
    yVariance = ( 1.0 / ( len( y ) - 1 ) ) * numpy.nansum( ( y - yMean ) ** 2 )
    
    CCC = ( 2 * covariance ) / ( xVariance + yVariance + ( xMean - yMean ) ** 2 )

    return CCC

CccScore = make_scorer( calculateCcc, greater_is_better=True )

def CccScoreTensorflow( x, y ):
    return( convert_to_tensor( calculateCcc( x, y ) ) )