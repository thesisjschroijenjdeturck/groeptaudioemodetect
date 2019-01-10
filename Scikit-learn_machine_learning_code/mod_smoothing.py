import numpy as np

# If length of inputSequenceArray < filterLength, this function
# will return an empty array!
def movingAverageFilter( inputSequenceArray, filterLength ):
    seqArr = inputSequenceArray.flatten()
    seqArrLen = seqArr.size    
    filteredSeqList = []

    for index in range( ( filterLength - 1 ), seqArrLen ):
        filteredSeqList.append( np.sum( seqArr[ ( index - ( filterLength - 1 ) ) : ( index + 1 ) ] ) / filterLength )

    return np.array( filteredSeqList )